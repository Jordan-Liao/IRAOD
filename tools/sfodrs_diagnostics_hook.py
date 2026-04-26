from __future__ import annotations

import os

from mmcv.runner import HOOKS, Hook
from mmdet.utils import get_root_logger


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


@HOOKS.register_module()
class SFODRSDiagnosticsHook(Hook):
    def __init__(
        self,
        *,
        stage: str,
        target_domain: str,
        use_labeled_source_in_adaptation: bool,
        cga_enabled: bool,
        cga_mode: str = "sfodrs",
        prompt_template: str = 'A SAR image of a {}',
        keep_label: bool = True,
        score_rule: str = "0.7*teacher + 0.3*clip_prob_orig",
    ) -> None:
        self.stage = str(stage)
        self.target_domain = str(target_domain)
        self.use_labeled_source_in_adaptation = bool(use_labeled_source_in_adaptation)
        self.cga_enabled = bool(cga_enabled)
        self.cga_mode = str(cga_mode)
        self.prompt_template = str(prompt_template)
        self.keep_label = bool(keep_label)
        self.score_rule = str(score_rule)

    def before_run(self, runner) -> None:
        logger = get_root_logger()

        # Also surface the runtime CGA env so logs are self-contained.
        env_cga_scorer = os.environ.get("CGA_SCORER", "").strip()
        env_templates = os.environ.get("CGA_TEMPLATES", "").strip()
        env_tau = os.environ.get("CGA_TAU", "").strip()
        env_expand = os.environ.get("CGA_EXPAND_RATIO", "").strip()

        logger.info("[SFOD-RS/RSAR] stage=%s", self.stage)
        logger.info(
            "[SFOD-RS/RSAR] use_labeled_source_in_adaptation=%s",
            str(self.use_labeled_source_in_adaptation),
        )
        logger.info("[SFOD-RS/RSAR] target_domain=%s", self.target_domain)
        logger.info("[SFOD-RS/RSAR] cga_enabled=%s cga_mode=%s", str(self.cga_enabled), self.cga_mode)
        logger.info('[SFOD-RS/RSAR] prompt_template="%s"', self.prompt_template)
        logger.info("[SFOD-RS/RSAR] pseudo_label_fusion: keep_label=%s score_rule=%s", str(self.keep_label), self.score_rule)
        if any([env_cga_scorer, env_templates, env_tau, env_expand]):
            logger.info(
                "[SFOD-RS/RSAR] CGA env: CGA_SCORER=%s CGA_TEMPLATES=%s CGA_TAU=%s CGA_EXPAND_RATIO=%s",
                env_cga_scorer or "(unset)",
                env_templates or "(unset)",
                env_tau or "(unset)",
                env_expand or "(unset)",
            )

