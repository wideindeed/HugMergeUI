"""Five independent, topically-disjoint held-out passages used to compute
mean perplexity. A single short passage (the original Phase 5 approach) is
too noisy to anchor a real correlation study - averaging loss across several
unrelated domains reduces sample-to-sample variance in the ground-truth
quality signal itself.
"""

EVAL_TEXTS = [
    """
    The history of the compass begins in ancient China, where lodestones were
    first used to determine direction. By the Han dynasty, Chinese scientists
    had discovered that a piece of lodestone, when suspended freely, would
    align itself along a north-south axis. This property was gradually
    refined into a navigational instrument over the following centuries,
    eventually spreading along trade routes to the Islamic world and then to
    Europe, where it became indispensable to long-distance maritime trade.
    """.strip(),
    """
    Photosynthesis converts light energy into chemical energy stored in
    glucose. Inside the chloroplast, light-dependent reactions split water
    molecules, releasing oxygen as a byproduct and generating ATP and NADPH.
    These energy carriers then power the Calvin cycle, a light-independent
    process that fixes atmospheric carbon dioxide into three-carbon sugars.
    Nearly all food chains on Earth ultimately trace their energy back to
    this process occurring in plants, algae, and some bacteria.
    """.strip(),
    """
    Risotto is traditionally made by toasting short-grain rice in butter or
    oil before gradually adding warm stock, one ladle at a time, stirring
    continuously to release the rice's starch. This slow addition of liquid
    is what gives risotto its characteristic creamy texture without any
    cream being added at all. Parmesan and a final knob of butter are
    typically stirred in off the heat, a step Italian cooks call the
    mantecatura, to finish the dish with a glossy sheen.
    """.strip(),
    """
    The 2008 financial crisis was triggered in large part by the collapse of
    the U.S. housing bubble and the widespread securitization of subprime
    mortgages into complex financial instruments. When default rates began
    to rise, the value of mortgage-backed securities collapsed, exposing
    banks and investment firms with heavy exposure to enormous losses. The
    resulting credit freeze spread rapidly across global markets, prompting
    coordinated central bank interventions and a prolonged global recession.
    """.strip(),
    """
    Coral reefs are built over thousands of years by colonies of tiny
    animals called polyps, which secrete calcium carbonate skeletons that
    accumulate into vast reef structures. These reefs host an extraordinary
    density of marine biodiversity despite covering less than one percent of
    the ocean floor. Rising sea temperatures cause coral bleaching, in which
    stressed polyps expel the symbiotic algae that supply most of their
    energy, often leading to mass die-offs if warm conditions persist.
    """.strip(),
]
