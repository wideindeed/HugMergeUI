export interface TourStep {
  id: string
  title: string
  text: string
}

export const TOUR_STEPS: TourStep[] = [
  {
    id: 'examples',
    title: 'Try an example',
    text: 'This basically lets you skip typing anything. Click a card and it loads a real pair of already-fine-tuned models and checks them automatically, the fastest way to see the tool do something.',
  },
  {
    id: 'editor',
    title: 'Merge config',
    text: 'This is the actual recipe mergekit would use to combine models: which models, how much weight each gets, and the merge method. You can edit it by hand to try your own model pair.',
  },
  {
    id: 'architecture',
    title: 'Architecture check',
    text: "Before doing anything slow, this just checks the models are actually compatible shapes: same architecture family, same layer count. If they're not, merging them wouldn't even work.",
  },
  {
    id: 'picker',
    title: 'Conflict score inputs',
    text: "Pick the 'base' model (the common starting point both fine-tunes came from) and the two models being compared, then hit Score. This is what actually kicks off the analysis.",
  },
  {
    id: 'results',
    title: 'Results',
    text: "This shows drift_magnitude, layer by layer: how far apart the two models' weight changes are. Green means low drift, red means high drift. It's validated against real merge quality at 1.5B+ params (see the caveat text below the chart for the specifics and what's still unproven).",
  },
]
