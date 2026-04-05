"""DEMAgent and ADEMAgent: integrated inference agents.

DEMAgent performs perception (D-step) only.
ADEMAgent additionally performs action updates (A-step).
Both agents optionally support the E-step (parameter inference).
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
import jax.numpy as jnp

from .model import DEMModel
from .inference import DStep, compute_vfe
from .action import ActionUpdate
from .estep import EStep


@dataclass
class DEMState:
    """Current state of a DEMAgent.

    Attributes:
        mu_x_tilde: Posterior mean over generalized states, shape (n_x*n_order,).
        mu_v_tilde: Posterior mean over generalized causes, shape (n_v*n_order,).
        vfe_history: List of VFE values recorded at each step.
        t: Current time.
        params: Current parameter estimate θ (None if E-step not used).
        params_history: List of parameter vectors recorded at each E-step.
    """

    mu_x_tilde: jnp.ndarray
    mu_v_tilde: jnp.ndarray
    vfe_history: List[float] = field(default_factory=list)
    t: float = 0.0
    params: Optional[jnp.ndarray] = None
    params_history: List[jnp.ndarray] = field(default_factory=list)


class DEMAgent:
    """DEM Agent: perception via D-step VFE minimization.

    Performs state inference by iteratively running the D-step (gradient descent
    on VFE) as observations arrive. Optionally also performs the E-step
    (parameter inference) at a specified frequency.

    Args:
        model: DEMModel specifying the generative model.
        kappa_mu: Learning rate for state inference (default 1.0).
        dt: Integration time step (default 0.001).
        n_iter_per_step: Number of D-step Euler iterations per observation
            (default 32).
        use_d_operator: If True, include D*mu term in the D-step update
            (full DEM generalized motion). If False (default), use stable
            gradient descent without the D term.
        use_estep: If True, run E-step (parameter inference) alongside the
            D-step (default False).
        kappa_p: Learning rate for parameter update in E-step (default 0.01).
        n_estep_per_obs: Number of E-step gradient descent iterations per
            observation (default 1).

    Example:
        >>> model = LinearDEMModel(A, C)
        >>> agent = DEMAgent(model)
        >>> mu_x0 = jnp.zeros(model.dim_x_tilde)
        >>> mu_v0 = jnp.zeros(model.dim_v_tilde)
        >>> state = DEMState(mu_x0, mu_v0)
        >>> for y_tilde in observations:
        ...     state = agent.step(state, y_tilde)
    """

    def __init__(
        self,
        model: DEMModel,
        kappa_mu: float = 1.0,
        dt: float = 0.001,
        n_iter_per_step: int = 32,
        use_d_operator: bool = False,
        use_estep: bool = False,
        kappa_p: float = 0.01,
        n_estep_per_obs: int = 1,
    ) -> None:
        self.model = model
        self.use_estep = use_estep
        self.n_estep_per_obs = n_estep_per_obs
        self.d_step = DStep(model, kappa_mu=kappa_mu, dt=dt, n_iter=n_iter_per_step,
                            use_d_operator=use_d_operator)
        if use_estep:
            self.e_step = EStep(model, kappa_p=kappa_p)
        else:
            self.e_step = None

    def init_state(
        self,
        mu_x0: Optional[jnp.ndarray] = None,
        mu_v0: Optional[jnp.ndarray] = None,
        params0: Optional[jnp.ndarray] = None,
    ) -> DEMState:
        """Initialize agent state.

        Args:
            mu_x0: Initial state mean, shape (n_x*n_order,).
                   Defaults to zeros.
            mu_v0: Initial cause mean, shape (n_v*n_order,).
                   Defaults to zeros.
            params0: Initial parameter vector θ (optional). If None and
                     use_estep is True, uses model.params.

        Returns:
            Initialized DEMState.
        """
        if mu_x0 is None:
            mu_x0 = jnp.zeros(self.model.dim_x_tilde)
        if mu_v0 is None:
            mu_v0 = jnp.zeros(self.model.dim_v_tilde)
        if params0 is None and self.use_estep and self.model.params is not None:
            params0 = jnp.asarray(self.model.params)
        return DEMState(
            mu_x_tilde=mu_x0,
            mu_v_tilde=mu_v0,
            params=params0,
            params_history=[params0] if params0 is not None else [],
        )

    def step(
        self,
        state: DEMState,
        y_tilde: jnp.ndarray,
    ) -> DEMState:
        """Perform one D-step (and optionally E-step) update given a new observation.

        When use_estep is True, runs the E-step after the D-step using the
        current observation to update the parameter estimate θ.

        Args:
            state: Current DEMState.
            y_tilde: Generalized observation vector, shape (n_y*n_order,).

        Returns:
            Updated DEMState with new mu_x_tilde, mu_v_tilde, and appended VFE.
            If use_estep is True, also updates params and params_history.
        """
        new_mu_x, new_mu_v, vfe = self.d_step.run(
            state.mu_x_tilde, state.mu_v_tilde, y_tilde
        )
        new_history = state.vfe_history + [vfe]

        new_params = state.params
        new_params_history = state.params_history

        if self.use_estep and self.e_step is not None and new_params is not None:
            # E-step: update parameters using current observation
            new_params = self.e_step.run(
                mu_x_sequence=[new_mu_x],
                mu_v_sequence=[new_mu_v],
                y_sequence=[y_tilde],
                params=new_params,
                n_iter=self.n_estep_per_obs,
            )
            new_params_history = state.params_history + [new_params]

        return DEMState(
            mu_x_tilde=new_mu_x,
            mu_v_tilde=new_mu_v,
            vfe_history=new_history,
            t=state.t + self.d_step.dt * self.d_step.n_iter,
            params=new_params,
            params_history=new_params_history,
        )

    def run(
        self,
        y_tilde_sequence: List[jnp.ndarray],
        mu_x0: Optional[jnp.ndarray] = None,
        mu_v0: Optional[jnp.ndarray] = None,
        params0: Optional[jnp.ndarray] = None,
    ) -> Tuple[DEMState, List[jnp.ndarray]]:
        """Run the agent over a sequence of observations.

        Args:
            y_tilde_sequence: List of generalized observation vectors.
            mu_x0: Initial state mean (optional, defaults to zeros).
            mu_v0: Initial cause mean (optional, defaults to zeros).
            params0: Initial parameter vector θ (optional, defaults to
                     model.params if use_estep is True).

        Returns:
            Tuple of (final DEMState, list of mu_x_tilde at each step).
        """
        state = self.init_state(mu_x0, mu_v0, params0)
        mu_x_history = [state.mu_x_tilde]

        for y_tilde in y_tilde_sequence:
            state = self.step(state, y_tilde)
            mu_x_history.append(state.mu_x_tilde)

        return state, mu_x_history


@dataclass
class ADEMState(DEMState):
    """Current state of an ADEMAgent, extending DEMState with action.

    Attributes:
        a: Current action vector, shape (n_a,).
    """

    a: jnp.ndarray = field(default_factory=lambda: jnp.zeros(1))  # type: ignore[assignment]


class ADEMAgent:
    """ADEM Agent: perception (D-step) + action (A-step) + optional E-step.

    Extends DEMAgent with an action update module that minimizes VFE
    by modifying actions that influence observations. Optionally also
    performs the E-step (parameter inference).

    Args:
        model: DEMModel specifying the generative model.
        g_action: Observation function including action:
                  g_action(x_tilde, v_tilde, a, params) -> y_tilde.
                  If None, action does not affect observations.
        kappa_mu: Learning rate for state inference (default 1.0).
        kappa_a: Learning rate for action update (default 1.0).
        dt: Integration time step (default 0.001).
        n_iter_per_step: Number of D-step iterations per observation (default 32).
        n_action_iter: Number of action update iterations per observation (default 4).
        use_d_operator: If True, include D*mu term in the D-step (default False).
        use_estep: If True, run E-step (parameter inference) alongside
            the D-step (default False).
        kappa_p: Learning rate for parameter update in E-step (default 0.01).
        n_estep_per_obs: Number of E-step iterations per observation (default 1).
    """

    def __init__(
        self,
        model: DEMModel,
        g_action: Optional[Callable] = None,
        kappa_mu: float = 1.0,
        kappa_a: float = 1.0,
        dt: float = 0.001,
        n_iter_per_step: int = 32,
        n_action_iter: int = 4,
        use_d_operator: bool = False,
        use_estep: bool = False,
        kappa_p: float = 0.01,
        n_estep_per_obs: int = 1,
    ) -> None:
        self.model = model
        self.use_estep = use_estep
        self.n_estep_per_obs = n_estep_per_obs
        self.d_step = DStep(model, kappa_mu=kappa_mu, dt=dt, n_iter=n_iter_per_step,
                            use_d_operator=use_d_operator)
        self.action_update = ActionUpdate(
            model, g_action=g_action, kappa_a=kappa_a, dt=dt
        )
        self.n_action_iter = n_action_iter
        if use_estep:
            self.e_step = EStep(model, kappa_p=kappa_p)
        else:
            self.e_step = None

    def init_state(
        self,
        mu_x0: Optional[jnp.ndarray] = None,
        mu_v0: Optional[jnp.ndarray] = None,
        a0: Optional[jnp.ndarray] = None,
        n_a: int = 1,
        params0: Optional[jnp.ndarray] = None,
    ) -> ADEMState:
        """Initialize agent state.

        Args:
            mu_x0: Initial state mean (optional, defaults to zeros).
            mu_v0: Initial cause mean (optional, defaults to zeros).
            a0: Initial action (optional, defaults to zeros).
            n_a: Action dimension (used if a0 is None).
            params0: Initial parameter vector θ (optional). If None and
                     use_estep is True, uses model.params.

        Returns:
            Initialized ADEMState.
        """
        if mu_x0 is None:
            mu_x0 = jnp.zeros(self.model.dim_x_tilde)
        if mu_v0 is None:
            mu_v0 = jnp.zeros(self.model.dim_v_tilde)
        if a0 is None:
            a0 = jnp.zeros(n_a)
        if params0 is None and self.use_estep and self.model.params is not None:
            params0 = jnp.asarray(self.model.params)
        return ADEMState(
            mu_x_tilde=mu_x0,
            mu_v_tilde=mu_v0,
            a=a0,
            params=params0,
            params_history=[params0] if params0 is not None else [],
        )

    def step(
        self,
        state: ADEMState,
        y_tilde: jnp.ndarray,
    ) -> ADEMState:
        """Perform one D-step (inference) + A-step (action) + optional E-step.

        Args:
            state: Current ADEMState.
            y_tilde: Generalized observation, shape (n_y*n_order,).

        Returns:
            Updated ADEMState.
        """
        # D-step: inference
        new_mu_x, new_mu_v, vfe = self.d_step.run(
            state.mu_x_tilde, state.mu_v_tilde, y_tilde
        )

        # A-step: action update
        a = state.a
        for _ in range(self.n_action_iter):
            a = self.action_update.step(a, new_mu_x, new_mu_v, y_tilde)

        new_history = state.vfe_history + [vfe]

        new_params = state.params
        new_params_history = state.params_history

        if self.use_estep and self.e_step is not None and new_params is not None:
            new_params = self.e_step.run(
                mu_x_sequence=[new_mu_x],
                mu_v_sequence=[new_mu_v],
                y_sequence=[y_tilde],
                params=new_params,
                n_iter=self.n_estep_per_obs,
            )
            new_params_history = state.params_history + [new_params]

        return ADEMState(
            mu_x_tilde=new_mu_x,
            mu_v_tilde=new_mu_v,
            vfe_history=new_history,
            t=state.t + self.d_step.dt * self.d_step.n_iter,
            a=a,
            params=new_params,
            params_history=new_params_history,
        )

    def run(
        self,
        y_tilde_sequence: List[jnp.ndarray],
        mu_x0: Optional[jnp.ndarray] = None,
        mu_v0: Optional[jnp.ndarray] = None,
        a0: Optional[jnp.ndarray] = None,
        n_a: int = 1,
        params0: Optional[jnp.ndarray] = None,
    ) -> Tuple[ADEMState, List[jnp.ndarray], List[jnp.ndarray]]:
        """Run the agent over a sequence of observations.

        Args:
            y_tilde_sequence: List of generalized observation vectors.
            mu_x0: Initial state mean (optional).
            mu_v0: Initial cause mean (optional).
            a0: Initial action (optional).
            n_a: Action dimension (used if a0 is None).
            params0: Initial parameter vector θ (optional).

        Returns:
            Tuple of (final ADEMState, mu_x history, action history).
        """
        state = self.init_state(mu_x0, mu_v0, a0, n_a, params0)
        mu_x_history = [state.mu_x_tilde]
        a_history = [state.a]

        for y_tilde in y_tilde_sequence:
            state = self.step(state, y_tilde)
            mu_x_history.append(state.mu_x_tilde)
            a_history.append(state.a)

        return state, mu_x_history, a_history
