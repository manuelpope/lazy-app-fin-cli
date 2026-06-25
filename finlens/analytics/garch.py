import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import DataScaleWarning

warnings.filterwarnings("ignore", category=DataScaleWarning)


@dataclass
class GARCHResult:
    omega: float
    alpha: float
    gamma: float
    beta: float
    persistence: float
    conditional_vol: float
    annualized_vol: float
    signal: str


def fit_gjr_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
) -> GARCHResult:
    r = returns.dropna()

    try:
        model = arch_model(
            r,
            mean="zero",
            vol="GARCH",
            p=p,
            o=1,
            q=q,
            dist="t",
        )
        result = model.fit(disp="off", show_warning=False)

        omega = float(result.params.get("omega", 0))
        alpha = float(result.params.get("alpha[1]", 0))
        gamma = float(result.params.get("gamma[1]", 0))
        beta = float(result.params.get("beta[1]", 0))

        persistence = alpha + gamma / 2 + beta
        cond_vol = result.conditional_volatility.iloc[-1]
        ann_vol = cond_vol * float(np.sqrt(252))

        if ann_vol < 0.15:
            signal = "LOW VOL"
        elif ann_vol > 0.30:
            signal = "HIGH VOL"
        else:
            signal = "MED VOL"

        return GARCHResult(
            omega=round(omega, 8),
            alpha=round(alpha, 4),
            gamma=round(gamma, 4),
            beta=round(beta, 4),
            persistence=round(persistence, 4),
            conditional_vol=round(cond_vol, 4),
            annualized_vol=round(ann_vol, 4),
            signal=signal,
        )
    except Exception as e:
        return GARCHResult(
            omega=0,
            alpha=0,
            gamma=0,
            beta=0,
            persistence=0,
            conditional_vol=0,
            annualized_vol=0,
            signal="N/A",
        )
