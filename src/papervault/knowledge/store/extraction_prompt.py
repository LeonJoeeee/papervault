"""KS domain-tuned entity-extraction prompt (research variants V2 + V3, KS_RAG_SOTA_RESEARCH.md).

WHY: LightRAG's stock few-shot examples are a fiction scene / finance blurb / sports headline
whose <Output> blocks emit person/organization/equipment/category — so the extractor is TAUGHT
to pull people, orgs, and citation-shaped proper nouns, which is the root cause of KS's
off-ontology + citation-entity noise. KS only injected the 11-type ontology via addon_params
(a weak hint) while the EXAMPLES showed the opposite. This module replaces the examples with
space-physics / AI-for-science ones (V2) and adds explicit negative exclusions + an acronym
canonicalization rule to the system prompt (V3).

Mutates the module-level lightrag.prompt.PROMPTS dict, so call apply_ks_extraction_prompt()
BEFORE the extraction runs (before ainsert / get_graph builds). operate.py reads PROMPTS at
extraction time, .format()-substituting {tuple_delimiter}/{completion_delimiter}/{entity_types}/
{language} — so the strings below MUST contain no other literal braces.
"""
from __future__ import annotations

from lightrag import prompt as lr_prompt

# 11-type research ontology (lowercase = what LightRAG stores), kept in sync with graph.ENTITY_TYPES.
_TYPES = '["concept","method","phenomenon","dataset","instrument","mission","quantity","material","model","finding","application"]'

# --- V2: domain few-shot examples (exact LightRAG 4-field entity / 5-field relation format).
# Each deliberately CONTAINS an in-text citation + a math symbol and extracts NEITHER, and
# collapses an acronym (PINNs) into ONE full-name entity.
_EX_SEP = (
    "<Entity_types>\n" + _TYPES + "\n\n"
    "<Input Text>\n```\n"
    "Solar energetic particles (SEPs) accelerated during coronal mass ejections propagate through "
    "the heliosphere, where their transport is governed by the parallel diffusion coefficient. "
    "Observations from the Parker Solar Probe show that the particle mean free path increases with "
    "radial distance, consistent with quasi-linear theory; the parallel diffusion coefficient "
    "$\\kappa_\\parallel$ scales roughly as $\\kappa_\\parallel \\propto r^1.17$ in this regime. "
    "Cane et al. (2003) reported a correlation "
    "between the onset delay and the ambient solar wind density, later refined using SOHO measurements. "
    "The focused transport equation, which balances streaming and pitch-angle scattering, reproduces "
    "the observed intensity-time profiles for most gradual events.\n```\n\n"
    "<Output>\n"
    "entity{tuple_delimiter}Solar Energetic Particles{tuple_delimiter}phenomenon{tuple_delimiter}Energetic charged particles (abbreviated SEPs) accelerated during solar eruptions that propagate through the heliosphere.\n"
    "entity{tuple_delimiter}Coronal Mass Ejection{tuple_delimiter}phenomenon{tuple_delimiter}A large eruption of plasma and magnetic field from the Sun that can accelerate solar energetic particles.\n"
    "entity{tuple_delimiter}Heliosphere{tuple_delimiter}phenomenon{tuple_delimiter}The region of space dominated by the solar wind through which solar energetic particles propagate.\n"
    "entity{tuple_delimiter}Parallel Diffusion Coefficient{tuple_delimiter}quantity{tuple_delimiter}A transport parameter governing particle diffusion along the magnetic field; observed to increase with radial distance.\n"
    "entity{tuple_delimiter}Parker Solar Probe{tuple_delimiter}mission{tuple_delimiter}A spacecraft whose observations show the particle mean free path increasing with radial distance.\n"
    "entity{tuple_delimiter}Quasi-Linear Theory{tuple_delimiter}method{tuple_delimiter}A theoretical framework for particle scattering, consistent with the observed radial trend of the mean free path.\n"
    "entity{tuple_delimiter}SOHO{tuple_delimiter}mission{tuple_delimiter}A solar observatory whose measurements refined the onset-delay versus solar-wind-density correlation.\n"
    "entity{tuple_delimiter}Focused Transport Equation{tuple_delimiter}model{tuple_delimiter}A transport model balancing streaming and pitch-angle scattering that reproduces SEP intensity-time profiles for gradual events.\n"
    "entity{tuple_delimiter}Onset Delay{tuple_delimiter}quantity{tuple_delimiter}The delay in solar-energetic-particle arrival, observed to correlate with ambient solar wind density.\n"
    "relation{tuple_delimiter}Solar Energetic Particles{tuple_delimiter}Coronal Mass Ejection{tuple_delimiter}acceleration, source{tuple_delimiter}Solar energetic particles are accelerated during coronal mass ejections.\n"
    "relation{tuple_delimiter}Parallel Diffusion Coefficient{tuple_delimiter}Solar Energetic Particles{tuple_delimiter}transport, governing parameter{tuple_delimiter}The parallel diffusion coefficient governs the transport of solar energetic particles.\n"
    "relation{tuple_delimiter}Focused Transport Equation{tuple_delimiter}Solar Energetic Particles{tuple_delimiter}modeling, intensity profiles{tuple_delimiter}The focused transport equation reproduces observed solar-energetic-particle intensity-time profiles.\n"
    "relation{tuple_delimiter}Parker Solar Probe{tuple_delimiter}Parallel Diffusion Coefficient{tuple_delimiter}observation, radial dependence{tuple_delimiter}Parker Solar Probe observations bear on the radial increase of the parallel diffusion coefficient.\n"
    "{completion_delimiter}\n"
)

_EX_PINN = (
    "<Entity_types>\n" + _TYPES + "\n\n"
    "<Input Text>\n```\n"
    "Physics-Informed Neural Networks (PINNs) solve partial differential equations by embedding the "
    "governing equations into the loss function. The total loss combines a data term and a residual "
    "term evaluated at collocation points. Raissi et al. (2019) introduced the framework, and later "
    "work showed that the gradient of the loss with respect to the network parameters can become "
    "ill-conditioned for stiff problems. Domain-decomposition variants such as XPINNs partition the "
    "domain to improve scalability. The approach has been applied to the heat equation and to inverse "
    "problems in fluid dynamics.\n```\n\n"
    "<Output>\n"
    "entity{tuple_delimiter}Physics-Informed Neural Network{tuple_delimiter}method{tuple_delimiter}A neural-network method (abbreviated PINN) that solves partial differential equations by embedding the governing equations into the training loss.\n"
    "entity{tuple_delimiter}Partial Differential Equation{tuple_delimiter}concept{tuple_delimiter}The governing equations that physics-informed neural networks are designed to solve.\n"
    "entity{tuple_delimiter}Residual Loss{tuple_delimiter}quantity{tuple_delimiter}The loss component evaluated at collocation points that enforces the governing equations during training.\n"
    "entity{tuple_delimiter}Collocation Points{tuple_delimiter}concept{tuple_delimiter}Sampling points in the domain where the residual loss of a physics-informed neural network is evaluated.\n"
    "entity{tuple_delimiter}XPINNs{tuple_delimiter}method{tuple_delimiter}A domain-decomposition variant of physics-informed neural networks that partitions the domain to improve scalability.\n"
    "entity{tuple_delimiter}Domain Decomposition{tuple_delimiter}method{tuple_delimiter}A strategy that partitions the computational domain to improve the scalability of neural-network PDE solvers.\n"
    "entity{tuple_delimiter}Inverse Problem{tuple_delimiter}application{tuple_delimiter}A class of fluid-dynamics problems to which physics-informed neural networks have been applied.\n"
    "entity{tuple_delimiter}Stiff Problem{tuple_delimiter}concept{tuple_delimiter}A class of problems for which the loss gradient with respect to network parameters can become ill-conditioned.\n"
    "relation{tuple_delimiter}Physics-Informed Neural Network{tuple_delimiter}Partial Differential Equation{tuple_delimiter}solving, governing equations{tuple_delimiter}Physics-informed neural networks solve partial differential equations by embedding them in the loss.\n"
    "relation{tuple_delimiter}Physics-Informed Neural Network{tuple_delimiter}Residual Loss{tuple_delimiter}training, loss component{tuple_delimiter}The residual loss enforces the governing equations during training of a physics-informed neural network.\n"
    "relation{tuple_delimiter}XPINNs{tuple_delimiter}Domain Decomposition{tuple_delimiter}method, scalability{tuple_delimiter}XPINNs apply domain decomposition to improve the scalability of physics-informed neural networks.\n"
    "relation{tuple_delimiter}Residual Loss{tuple_delimiter}Collocation Points{tuple_delimiter}evaluation, sampling{tuple_delimiter}The residual loss is evaluated at collocation points.\n"
    "{completion_delimiter}\n"
)

_EX_GCR = (
    "<Entity_types>\n" + _TYPES + "\n\n"
    "<Input Text>\n```\n"
    "Galactic cosmic rays are modulated by the solar wind as they propagate into the inner heliosphere, "
    "producing an anti-correlation between cosmic-ray intensity and solar activity. The AMS-02 detector "
    "aboard the International Space Station has measured the proton and helium spectra with high precision, "
    "while the PAMELA experiment provided earlier measurements of the positron fraction. These data "
    "constrain the local interstellar spectrum used in solar modulation models. The force-field "
    "approximation offers a single-parameter description of modulation, although it breaks down at low "
    "rigidities.\n```\n\n"
    "<Output>\n"
    "entity{tuple_delimiter}Galactic Cosmic Rays{tuple_delimiter}phenomenon{tuple_delimiter}High-energy charged particles from outside the solar system, modulated by the solar wind in the inner heliosphere.\n"
    "entity{tuple_delimiter}Solar Modulation{tuple_delimiter}phenomenon{tuple_delimiter}The suppression of galactic cosmic-ray intensity by the solar wind, anti-correlated with solar activity.\n"
    "entity{tuple_delimiter}AMS-02{tuple_delimiter}instrument{tuple_delimiter}A particle detector aboard the International Space Station that measures cosmic-ray proton and helium spectra with high precision.\n"
    "entity{tuple_delimiter}PAMELA{tuple_delimiter}instrument{tuple_delimiter}A space experiment that provided early measurements of the cosmic-ray positron fraction.\n"
    "entity{tuple_delimiter}Positron Fraction{tuple_delimiter}quantity{tuple_delimiter}The ratio of positrons to total electrons-plus-positrons in cosmic rays, measured by PAMELA.\n"
    "entity{tuple_delimiter}Local Interstellar Spectrum{tuple_delimiter}quantity{tuple_delimiter}The cosmic-ray spectrum outside the heliosphere, constrained by detector data and used in modulation models.\n"
    "entity{tuple_delimiter}Force-Field Approximation{tuple_delimiter}model{tuple_delimiter}A single-parameter description of solar modulation that breaks down at low rigidities.\n"
    "relation{tuple_delimiter}Galactic Cosmic Rays{tuple_delimiter}Solar Modulation{tuple_delimiter}modulation, anti-correlation{tuple_delimiter}Galactic cosmic rays undergo solar modulation, anti-correlated with solar activity.\n"
    "relation{tuple_delimiter}AMS-02{tuple_delimiter}Galactic Cosmic Rays{tuple_delimiter}measurement, spectra{tuple_delimiter}AMS-02 measures the proton and helium spectra of galactic cosmic rays.\n"
    "relation{tuple_delimiter}Force-Field Approximation{tuple_delimiter}Solar Modulation{tuple_delimiter}modeling, single-parameter{tuple_delimiter}The force-field approximation describes solar modulation with a single parameter.\n"
    "relation{tuple_delimiter}Local Interstellar Spectrum{tuple_delimiter}Solar Modulation{tuple_delimiter}input, modeling constraint{tuple_delimiter}The local interstellar spectrum constrains solar modulation models.\n"
    "{completion_delimiter}\n"
)

KS_EXAMPLES = [_EX_SEP, _EX_PINN, _EX_GCR]

# --- V3: negative exclusions + acronym rule, inserted before the system prompt's ---Examples--- block.
_EXCLUSIONS = """9.  **Exclusions (do NOT extract any of these as entities):**
    *   In-text bibliographic citations / author-year reference strings (e.g. `Cane et al. (2003)`, `Raissi (2019)`). Extract the concept, method, or finding being discussed — never the citation itself.
    *   Isolated mathematical symbols, variable names, subscripted quantities, or equation/figure/table labels (e.g. `S_m`, `x_0`, `Equation 9`). Extract the named quantity it denotes (e.g. `diffusion coefficient`), not the symbol.
    *   Generic type-words used with no specific name (a bare `Model`, `Method`, `Study`, `Approach`).
    *   Individual people, author names, institutions, universities, journals, funding agencies, or scientific collaborations treated as organizations.

10. **Acronym canonicalization:** When a concept is introduced with an acronym in parentheses (e.g. `Physics-Informed Neural Networks (PINNs)`), emit a SINGLE entity using the full canonical name and mention the acronym in its description — do NOT create separate entities for the acronym and its expansion.

11. **Capture the scientific substance in the description:** Each entity_description must preserve the *specific* content the text states about the entity, not a generic gloss. When the text gives them, include: quantitative facts (values, ranges, units, scaling, e.g. 'increases with radial distance'), the conditions or regime in which it holds, the mechanism, and the DIRECTION of any effect (increases/decreases, correlated/anti-correlated, governs/constrains). Prefer 'A transport parameter governing parallel diffusion; observed to increase with radial distance and consistent with quasi-linear theory' over 'A parameter related to particle transport.' Likewise, each relationship_description must state HOW the entities are related (the mechanism or quantitative dependence), not merely that they are related.

---Examples---"""


# --- V3: closed-ontology replacement for LightRAG's `Other` catch-all (system prompt item 1).
# The stock entity_type bullet ends with an escape hatch ("classify it as `Other`") that
# re-admits exactly the off-ontology proper-noun noise the 11-type ontology exists to drop
# (LightRAG lowercases entity_type on store, so `Other` lands as a real `other` node, not dropped).
# Replace it with a CLOSED-list + DROP instruction. Anchored on the exact vendored sentence.
_OTHER_CATCHALL = (
    "Categorize the entity using one of the following types: `{entity_types}`. "
    "If none of the provided entity types apply, do not add new entity type and classify it as `Other`."
)
_CLOSED_ONTOLOGY = (
    "Categorize the entity using EXACTLY one of the following types: `{entity_types}`. "
    "This list is CLOSED — never invent a new type and never use `Other`/`Unknown`. "
    "If a candidate entity does not clearly fit one of these types, DO NOT extract it at all "
    "(omit it entirely rather than forcing a catch-all)."
)


def apply_ks_extraction_prompt(examples: bool = True, exclusions: bool = True) -> None:
    """Mutate lightrag.prompt.PROMPTS in place. Idempotent. Call before extraction."""
    if exclusions:
        sysp = lr_prompt.PROMPTS["entity_extraction_system_prompt"]
        if "Exclusions (do NOT extract" not in sysp and "\n---Examples---" in sysp:
            sysp = sysp.replace("\n---Examples---", "\n" + _EXCLUSIONS, 1)
        # Drop the `Other` catch-all: close the ontology so off-ontology candidates are omitted.
        if _OTHER_CATCHALL in sysp:
            sysp = sysp.replace(_OTHER_CATCHALL, _CLOSED_ONTOLOGY, 1)
        lr_prompt.PROMPTS["entity_extraction_system_prompt"] = sysp
    if examples:
        lr_prompt.PROMPTS["entity_extraction_examples"] = list(KS_EXAMPLES)
