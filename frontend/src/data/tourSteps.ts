export interface TourStep {
  id: string
  title: string
  text: string
}

export const TOUR_STEPS: TourStep[] = [
  {
    id: 'examples',
    title: 'Try an example',
    text: 'This basically lets you skip typing anything. Click a card and it loads a real pair of already-fine-tuned models and checks them automatically — the fastest way to see the tool do something.',
  },
  {
    id: 'editor',
    title: 'Merge config',
    text: 'This is the actual recipe mergekit would use to combine models — which models, how much weight each gets, and the merge method. You can edit it by hand to try your own model pair.',
  },
  {
    id: 'architecture',
    title: 'Architecture check',
    text: "Before doing anything slow, this just checks the models are actually compatible shapes — same architecture family, same layer count. If they're not, merging them wouldn't even work.",
  },
  {
    id: 'picker',
    title: 'Conflict score inputs',
    text: "Pick the 'base' model (the common starting point both fine-tunes came from) and the two models being compared, then hit Score. This is what actually kicks off the analysis.",
  },
  {
    id: 'results',
    title: 'Results',
    text: "This shows how much the two models' changes agree or fight, layer by layer. Green = they moved weights the same direction, red = they fought each other. It's a heuristic signal, not a proven quality score — see the caveat text for why.",
  },
]
