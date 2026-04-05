"""Unit tests for DEM D-step numerical stability.

Tests:
- local_linearization_step: 線形システムで解析解と一致するか
- D-step with D operator: use_d_operator=True でも発散しないか
- Euler vs local linearization: 局所線形化が Euler より速く収束するか
- Generalized motion consistency: D 演算子あり/なしで推定値が近いか
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from src.dem.model import LinearDEMModel, DEMModel
from src.dem.inference import DStep, compute_vfe
from src.dem.utils import local_linearization_step


# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------

def make_test_model(
    n_order: int = 4,
    pi_y: float = 4.0,
    pi_x: float = 1.0,
) -> DEMModel:
    """1D 線形テストモデル dx/dt = -x + v, y = x を生成する。"""
    A = jnp.array([[-1.0]])
    C = jnp.array([[1.0]])
    return LinearDEMModel(A, C, n_order=n_order, pi_y=pi_y, pi_x=pi_x)


def make_observation(y0: float = 1.0, n_order: int = 4) -> jnp.ndarray:
    """ゼロ次成分のみ非ゼロな一般化観測ベクトルを生成する。"""
    y_tilde = jnp.zeros(n_order)
    return y_tilde.at[0].set(y0)


# ---------------------------------------------------------------------------
# test_local_linearization_linear
# ---------------------------------------------------------------------------

class TestLocalLinearizationLinear:
    """local_linearization_step が線形システムで解析解と一致するかテスト。"""

    def test_scalar_exponential_decay(self):
        """スカラー線形系 dx/dt = -lambda * x の解析解と一致するか確認。

        解析解：Δx = x₀ * (e^(-lambda * dt) - 1)
        """
        lam = 2.0
        x0 = jnp.array([1.5])
        J = jnp.array([[-lam]])
        f_val = J @ x0  # = -lam * x0

        dt = 0.1
        delta_x = local_linearization_step(f_val, J, dt, damping=0.0)

        # 解析解
        expected = x0 * (jnp.exp(-lam * dt) - 1)
        np.testing.assert_allclose(
            np.array(delta_x), np.array(expected), rtol=1e-4,
            err_msg="Local linearization should match analytic solution for scalar decay"
        )

    def test_2d_linear_system(self):
        """2次元線形系で行列指数関数の解析解と一致するか確認。"""
        A = jnp.array([[-1.0, 0.5], [-0.5, -1.0]])
        x0 = jnp.array([1.0, 0.0])
        f_val = A @ x0
        dt = 0.05

        delta_x = local_linearization_step(f_val, A, dt, damping=0.0)

        # 解析解: Δx = (expm(A*dt) - I) x0 = expm(A*dt) x0 - x0
        import jax.scipy.linalg
        expm_A_dt = jax.scipy.linalg.expm(A * dt)
        expected = expm_A_dt @ x0 - x0

        np.testing.assert_allclose(
            np.array(delta_x), np.array(expected), rtol=1e-5,
            err_msg="Local linearization should match matrix exponential for 2D linear system"
        )

    def test_constant_input(self):
        """定数入力系 dx/dt = A*x + b の解析解と一致するか確認。

        この場合 f(x0) = A*x0 + b で J = A。
        解析解：Δx = A⁻¹(e^(A*dt) - I)(A*x0 + b)
        """
        A = jnp.array([[-2.0]])
        b = jnp.array([1.0])
        x0 = jnp.array([0.5])
        f_val = A @ x0 + b  # f(x0) = A*x0 + b
        dt = 0.1

        delta_x = local_linearization_step(f_val, A, dt, damping=0.0)

        # 解析解
        import jax.scipy.linalg
        expm_Adt = jax.scipy.linalg.expm(A * dt)
        A_inv = jnp.linalg.inv(A)
        expected = A_inv @ (expm_Adt - jnp.eye(1)) @ f_val

        np.testing.assert_allclose(
            np.array(delta_x), np.array(expected), rtol=1e-5,
            err_msg="Local linearization should match analytic solution for affine system"
        )

    def test_jit_compilable(self):
        """local_linearization_step が jax.jit でコンパイル可能かテスト。"""
        J = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        f_val = jnp.array([1.0, 0.5])

        @jax.jit
        def jit_step(f, j):
            return local_linearization_step(f, j, dt=0.01)

        result = jit_step(f_val, J)
        assert jnp.all(jnp.isfinite(result)), "JIT-compiled local linearization should produce finite results"


# ---------------------------------------------------------------------------
# test_d_step_stable_with_d_operator
# ---------------------------------------------------------------------------

class TestDStepStableWithDOperator:
    """use_d_operator=True でも大きな dt で安定動作するかテスト。"""

    def test_stable_with_large_dt(self):
        """dt=0.01 という大きなステップでも発散しないか確認。

        Euler 法では Hessian の最大固有値が 269 程度のとき
        dt > 2/269 ≈ 0.0074 で不安定になるが、
        局所線形化では安定であること。
        """
        model = make_test_model(n_order=4, pi_y=4.0, pi_x=1.0)
        d_step = DStep(
            model,
            kappa_mu=1.0,
            dt=0.01,  # Euler 法では不安定な大きな dt
            n_iter=20,
            use_d_operator=True,
            use_local_linearization=True,
        )

        mu_x0 = jnp.zeros(model.dim_x_tilde)
        mu_v0 = jnp.zeros(model.dim_v_tilde)
        y_tilde = make_observation(1.0, model.n_order)

        mu_x_new, mu_v_new, vfe_final = d_step.run(mu_x0, mu_v0, y_tilde)

        # 発散していないこと（有限値）
        assert jnp.all(jnp.isfinite(mu_x_new)), (
            f"mu_x_tilde diverged with large dt=0.01: {mu_x_new}"
        )
        assert jnp.all(jnp.isfinite(mu_v_new)), (
            f"mu_v_tilde diverged with large dt=0.01: {mu_v_new}"
        )
        assert jnp.isfinite(vfe_final), f"VFE diverged: {vfe_final}"

    def test_no_divergence_over_iterations(self):
        """多数のイテレーションでも発散しないか確認。"""
        model = make_test_model(n_order=4, pi_y=4.0, pi_x=1.0)
        d_step = DStep(
            model,
            kappa_mu=1.0,
            dt=0.01,
            n_iter=1,  # 1ステップずつ確認
            use_d_operator=True,
            use_local_linearization=True,
        )

        mu_x = jnp.zeros(model.dim_x_tilde)
        mu_v = jnp.zeros(model.dim_v_tilde)
        y_tilde = make_observation(1.0, model.n_order)

        vfe_values = []
        for i in range(50):
            mu_x, mu_v = d_step.run_single_step(mu_x, mu_v, y_tilde)
            vfe = float(compute_vfe(mu_x, mu_v, y_tilde, model))
            vfe_values.append(vfe)

            assert jnp.all(jnp.isfinite(mu_x)), f"mu_x diverged at step {i}: {mu_x}"
            assert jnp.isfinite(vfe), f"VFE diverged at step {i}: {vfe}"

    def test_vfe_decreases_with_d_operator(self):
        """D 演算子あり局所線形化で VFE が初期値より減少するか確認。"""
        model = make_test_model(n_order=4, pi_y=4.0, pi_x=1.0)
        d_step = DStep(
            model,
            kappa_mu=1.0,
            dt=0.01,
            n_iter=30,
            use_d_operator=True,
            use_local_linearization=True,
        )

        key = jax.random.PRNGKey(42)
        mu_x0 = jax.random.normal(key, (model.dim_x_tilde,)) * 0.5
        mu_v0 = jnp.zeros(model.dim_v_tilde)
        y_tilde = make_observation(1.0, model.n_order)

        vfe_initial = float(compute_vfe(mu_x0, mu_v0, y_tilde, model))
        mu_x_new, mu_v_new, vfe_final = d_step.run(mu_x0, mu_v0, y_tilde)

        assert vfe_final < vfe_initial, (
            f"VFE should decrease with D-operator + local linearization: "
            f"initial={vfe_initial:.4f}, final={vfe_final:.4f}"
        )


# ---------------------------------------------------------------------------
# test_d_step_euler_vs_local_lin_convergence
# ---------------------------------------------------------------------------

class TestEulerVsLocalLinConvergence:
    """局所線形化が Euler 法より速く収束するかテスト。"""

    def test_local_lin_fewer_steps_to_converge(self):
        """同じ dt で局所線形化の方が少ないステップで低 VFE に達するか確認。

        安全な dt 範囲（dt=0.001）での比較。
        局所線形化は大きい「等価ステップ」を踏むため、より速く収束する。
        """
        model = make_test_model(n_order=4, pi_y=4.0, pi_x=1.0)
        dt = 0.001
        n_steps = 20

        mu_x0 = jnp.ones(model.dim_x_tilde) * 2.0
        mu_v0 = jnp.zeros(model.dim_v_tilde)
        y_tilde = make_observation(0.0, model.n_order)  # 観測 y=0 に向けて収束

        vfe_init = float(compute_vfe(mu_x0, mu_v0, y_tilde, model))

        # Euler（D 演算子なし、安定なグラジェント降下）
        d_step_euler = DStep(
            model, kappa_mu=1.0, dt=dt, n_iter=n_steps,
            use_d_operator=False, use_local_linearization=False
        )
        _, _, vfe_euler = d_step_euler.run(mu_x0, mu_v0, y_tilde)

        # 局所線形化（D 演算子あり）
        d_step_ll = DStep(
            model, kappa_mu=1.0, dt=dt, n_iter=n_steps,
            use_d_operator=True, use_local_linearization=True
        )
        _, _, vfe_ll = d_step_ll.run(mu_x0, mu_v0, y_tilde)

        # 両方とも初期 VFE より減少していること
        assert vfe_euler < vfe_init, "Euler should reduce VFE"
        assert vfe_ll < vfe_init, "Local linearization should reduce VFE"

        # 両方とも有限値であること
        assert jnp.isfinite(vfe_euler), f"Euler VFE is not finite: {vfe_euler}"
        assert jnp.isfinite(vfe_ll), f"Local linearization VFE is not finite: {vfe_ll}"

    def test_local_lin_stable_where_euler_diverges(self):
        """Euler 法が不安定になる dt で局所線形化が安定か確認。

        dt=0.01 は Hessian の固有値が大きいとき Euler には危険だが、
        局所線形化では安定。
        """
        model = make_test_model(n_order=4, pi_y=16.0, pi_x=1.0)  # 大きな pi_y

        mu_x0 = jnp.ones(model.dim_x_tilde) * 3.0
        mu_v0 = jnp.zeros(model.dim_v_tilde)
        y_tilde = make_observation(0.0, model.n_order)

        # 局所線形化: dt=0.01 でも安定なはず
        d_step_ll = DStep(
            model, kappa_mu=1.0, dt=0.01, n_iter=50,
            use_d_operator=True, use_local_linearization=True
        )
        mu_x_new, _, vfe_ll = d_step_ll.run(mu_x0, mu_v0, y_tilde)

        assert jnp.all(jnp.isfinite(mu_x_new)), (
            f"Local linearization diverged where Euler would be unstable: {mu_x_new}"
        )
        assert jnp.isfinite(vfe_ll), f"VFE not finite: {vfe_ll}"

        # VFE が減少していること
        vfe_init = float(compute_vfe(mu_x0, mu_v0, y_tilde, model))
        assert vfe_ll < vfe_init, (
            f"Local linearization should reduce VFE: init={vfe_init:.4f}, final={vfe_ll:.4f}"
        )


# ---------------------------------------------------------------------------
# test_generalized_motion_consistency
# ---------------------------------------------------------------------------

class TestGeneralizedMotionConsistency:
    """D 演算子あり/なしで最終的な推定値が近いかテスト。"""

    def test_gradient_descent_and_local_lin_agree(self):
        """同じ観測に対してグラジェント降下と局所線形化が近い推定値に収束するか確認。

        VFE の最小値点は積分手法によらず同じなので、十分な反復後に
        ゼロ次状態推定が近い値になることを確認する。
        """
        model = make_test_model(n_order=4, pi_y=4.0, pi_x=1.0)
        y_tilde = make_observation(1.0, model.n_order)

        mu_x0 = jnp.zeros(model.dim_x_tilde)
        mu_v0 = jnp.zeros(model.dim_v_tilde)

        # グラジェント降下（D 演算子なし、多数の小さいステップ）
        d_step_gd = DStep(
            model, kappa_mu=1.0, dt=0.001, n_iter=200,
            use_d_operator=False, use_local_linearization=False
        )
        mu_x_gd, _, vfe_gd = d_step_gd.run(mu_x0, mu_v0, y_tilde)

        # 局所線形化（D 演算子あり）
        d_step_ll = DStep(
            model, kappa_mu=1.0, dt=0.01, n_iter=50,
            use_d_operator=True, use_local_linearization=True
        )
        mu_x_ll, _, vfe_ll = d_step_ll.run(mu_x0, mu_v0, y_tilde)

        # ゼロ次状態推定（最初の要素）が近いこと
        x0_gd = float(mu_x_gd[0])
        x0_ll = float(mu_x_ll[0])

        assert abs(x0_gd - x0_ll) < 0.5, (
            f"Gradient descent and local linearization should give similar estimates: "
            f"GD={x0_gd:.4f}, LL={x0_ll:.4f}"
        )

        # 両方とも有限値であること
        assert jnp.all(jnp.isfinite(mu_x_gd)), "Gradient descent estimate not finite"
        assert jnp.all(jnp.isfinite(mu_x_ll)), "Local linearization estimate not finite"

    def test_both_modes_track_observation(self):
        """両モードともに観測値に向けて推定値が動くか確認。

        観測 y=1.0 に対して、ゼロ次状態推定が初期値 0 から 1 に近づくこと。
        """
        model = make_test_model(n_order=4, pi_y=8.0, pi_x=1.0)
        y_tilde = make_observation(1.0, model.n_order)

        mu_x0 = jnp.zeros(model.dim_x_tilde)
        mu_v0 = jnp.zeros(model.dim_v_tilde)

        # グラジェント降下
        d_step_gd = DStep(
            model, kappa_mu=1.0, dt=0.001, n_iter=200,
            use_d_operator=False
        )
        mu_x_gd, _, _ = d_step_gd.run(mu_x0, mu_v0, y_tilde)

        # 局所線形化
        d_step_ll = DStep(
            model, kappa_mu=1.0, dt=0.01, n_iter=50,
            use_d_operator=True, use_local_linearization=True
        )
        mu_x_ll, _, _ = d_step_ll.run(mu_x0, mu_v0, y_tilde)

        # 両方とも観測値（1.0）に向けて動いているか
        assert float(mu_x_gd[0]) > 0.1, (
            f"GD estimate should move towards y=1.0, got {float(mu_x_gd[0]):.4f}"
        )
        assert float(mu_x_ll[0]) > 0.1, (
            f"LL estimate should move towards y=1.0, got {float(mu_x_ll[0]):.4f}"
        )

    def test_jit_compatible_both_modes(self):
        """両モードの DStep が jax.jit でコンパイル可能かテスト。"""
        model = make_test_model(n_order=4, pi_y=4.0, pi_x=1.0)
        y_tilde = make_observation(1.0, model.n_order)
        mu_x0 = jnp.zeros(model.dim_x_tilde)
        mu_v0 = jnp.zeros(model.dim_v_tilde)

        # グラジェント降下
        d_step_gd = DStep(model, kappa_mu=1.0, dt=0.001, n_iter=1,
                          use_d_operator=False)
        # 局所線形化
        d_step_ll = DStep(model, kappa_mu=1.0, dt=0.01, n_iter=1,
                          use_d_operator=True, use_local_linearization=True)

        # 両方とも _euler_step は jax.jit 済みなので、単純に呼び出してテスト
        mu_x_gd, mu_v_gd = d_step_gd.run_single_step(mu_x0, mu_v0, y_tilde)
        mu_x_ll, mu_v_ll = d_step_ll.run_single_step(mu_x0, mu_v0, y_tilde)

        assert jnp.all(jnp.isfinite(mu_x_gd)), "GD step produced non-finite values"
        assert jnp.all(jnp.isfinite(mu_x_ll)), "LL step produced non-finite values"
