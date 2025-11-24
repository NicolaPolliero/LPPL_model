import numpy as np
import pandas as pd
from scipy.optimize import minimize


class ModelLPPLS:
    """
    Log-Periodic Power Law Singularity (LPPLS) model for bubble detection.

    This class fits the LPPLS function to a price time series using
    non-linear least squares optimization.

    Parameters
    ----------
    t : array-like
        Time points (e.g., in years).
    p : array-like
        Observed price time series (must be positive).

    Attributes
    ----------
    t : np.ndarray
        Time grid.
    p : np.ndarray
        Observed prices.
    logp : np.ndarray
        Log of observed prices.
    params : dict or None
        Dictionary of fitted parameters {A, B, C1, C2, m, omega, tc}.
    fitted : bool
        True if the model was successfully fitted.
    result : OptimizeResult
        Full optimization output (from scipy.optimize.minimize).
    """

    # ---------- Initialization ---------- #
    def __init__(self, t: np.ndarray, p: np.ndarray):
        self.t = np.asarray(t)
        self.p = np.asarray(p)
        self.logp = np.log(self.p)
        self.params = None
        self.fitted = False
        self.result = None
        self.calibration_date = None

    def set_calibration_date(self, date):
        self.calibration_date = pd.to_datetime(date)

    # ---------- Internal Utilities ---------- #
    def _design_matrix(self, tc: float, m: float, omega: float) -> np.ndarray:
        """Design matrix for linear regression part (internal)."""
        dt = np.maximum(tc - self.t, 1e-9)
        f = dt ** m
        g = f * np.cos(omega * np.log(dt))
        h = f * np.sin(omega * np.log(dt))
        return np.column_stack([np.ones_like(self.t), f, g, h])

    def _solve_linear_params(self, tc: float, m: float, omega: float):
        """Analytical OLS solution for A, B, C1, C2 (internal)."""
        X = self._design_matrix(tc, m, omega)
        beta, *_ = np.linalg.lstsq(X, self.logp, rcond=None)
        return beta  # A, B, C1, C2

    def _check_bounds(self, tc: float, m: float, omega: float) -> bool:
        """Stylized LPPL parameter constraints (Filimonov–Sornette)."""
        return (
            self.t[-1] < tc < self.t[-1] + 300 / 365
            and 0.1 <= m <= 0.9
            and 6 <= omega <= 13
        )

    def _sse(self, params):
        """Sum of squared errors (objective function)."""
        tc, m, omega = params
        if not self._check_bounds(tc, m, omega):
            return np.inf
        A, B, C1, C2 = self._solve_linear_params(tc, m, omega)
        y_pred = self.lppls(self.t, A, B, C1, C2, tc, m, omega)
        return np.sum((self.logp - y_pred) ** 2)

    # ---------- Public Methods ---------- #
    def lppls(self, t, A, B, C1, C2, tc, m, omega):
        """Compute the expected log-price for given parameters."""
        dt = np.maximum(tc - t, 1e-9)
        f = dt ** m
        return A + B * f + C1 * f * np.cos(omega * np.log(dt)) + C2 * f * np.sin(
            omega * np.log(dt)
        )

    def _check_qualified_fit(self, tc: float, m: float, omega: float) -> bool:
        """
        Qualified fit constraints (Filimonov–Sornette 2013).

        tc must lie within (-60, 252) days relative to last observation.
        m in (0, 1)
        omega in [2, 15]
        """
        tc_lower = -60 / 365.25
        tc_upper = 252 / 365.25
        tc_rel = tc - self.t[-1]

        return (
            tc_lower < tc_rel < tc_upper
            and 0 < m < 1
            and 2 <= omega <= 15
        )

    def fit(self, initial_guess, method: str = "Nelder-Mead", options=None):
        """
        Fit the LPPLS model by minimizing the sum of squared errors.
        """
        result = minimize(self._sse, initial_guess, method=method, options=options)
        self.result = result

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        tc, m, omega = result.x
        A, B, C1, C2 = self._solve_linear_params(tc, m, omega)
        self.params = {
            "tc": float(tc),
            "m": float(m),
            "omega": float(omega),
            "A": float(A),
            "B": float(B),
            "C1": float(C1),
            "C2": float(C2),
        }

        # Qualified fit?
        self.fitted = self._check_qualified_fit(tc, m, omega)
        return self

    def fit_multistart(self, n_runs: int = 10, tol: float = 0.01):
        """
        Robust multistart fitting:
        - Runs `fit()` many times with random initial guesses.
        - Keeps only candidates passing the qualified-fit check.
        - Early stops if RMSE < tol (default 0.01).

        Randomization:
        - m ~ U(0,1)
        - omega ~ U(1,50)
        - tc0 = t_last + U(0.01, 0.5)
        """
        best = None
        best_rmse = np.inf

        for _ in range(int(n_runs)):
            tc0 = self.t[-1] + float(np.random.uniform(0.01, 0.5))
            m0 = float(np.random.uniform(0.0, 1.0))
            omega0 = float(np.random.uniform(1.0, 50.0))

            try:
                candidate = ModelLPPLS(self.t, self.p)
                candidate.fit([tc0, m0, omega0])
            except Exception:
                continue

            if not candidate.fitted:
                continue

            pars = candidate.params
            y_pred = candidate.lppls(
                self.t,
                pars["A"], pars["B"], pars["C1"], pars["C2"],
                pars["tc"], pars["m"], pars["omega"]
            )
            rmse = np.sqrt(np.mean((self.logp - y_pred) ** 2))

            if np.isfinite(rmse) and rmse < best_rmse:
                best_rmse = rmse
                best = candidate

            # ---- EARLY STOP ----
            if rmse < tol:
                best_rmse = rmse
                best = candidate
                break

        if best is None:
            self.fitted = False
            raise RuntimeError("fit_multistart: no qualified fit found.")

        self.params = best.params
        self.result = best.result
        self.fitted = True
        return self

    def summary(self, calibration_date=None):
        """Return fitted parameters as a one-row DataFrame."""
        if not self.fitted:
            raise RuntimeError("Fit the model first.")

        A = self.params["A"]
        B = self.params["B"]
        C1 = self.params["C1"]
        C2 = self.params["C2"]
        tc = self.params["tc"]
        m = self.params["m"]
        omega = self.params["omega"]

        kappa = -B
        sign = 1 if kappa > 0 else -1

        row = {
            "calibration_date": self.calibration_date,
            "tc": tc,
            "A": A,
            "B": B,
            "C1": C1,
            "C2": C2,
            "m": m,
            "omega": omega,
            "kappa": kappa,
            "sign": sign,
        }

        return pd.DataFrame([row])
