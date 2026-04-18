import type { GraphData, Message } from '../types'

export const initialGraph: GraphData = {
  nodes: [
    { id: 'enzyme', label: 'Enzyme', type: 'concept', highlighted: false },
    { id: 'active-site', label: 'Active Site', type: 'concept', highlighted: false },
    { id: 'substrate', label: 'Substrate', type: 'concept', highlighted: false },
    { id: 'reaction-rate', label: 'Reaction Rate', type: 'concept', highlighted: false },
    { id: 'temperature', label: 'Temperature', type: 'concept', highlighted: false },
    { id: 'kinetics', label: 'Enzyme Kinetics', type: 'topic', highlighted: false },
    { id: 'inhibition', label: 'Inhibition', type: 'concept', highlighted: false },
    { id: 'denaturation', label: 'Denaturation', type: 'concept', highlighted: false },
    { id: 'thermodynamics', label: 'Thermodynamics', type: 'topic', highlighted: false },
    { id: 'catalyst', label: 'Catalyst', type: 'concept', highlighted: false },
    { id: 'ph', label: 'pH Level', type: 'concept', highlighted: false },
    { id: 'activation-energy', label: 'Activation Energy', type: 'concept', highlighted: false },
  ],
  edges: [
    { id: 'e1', source: 'enzyme', target: 'active-site', relation: 'HAS_PART', highlighted: false },
    { id: 'e2', source: 'active-site', target: 'substrate', relation: 'BINDS', highlighted: false },
    { id: 'e3', source: 'enzyme', target: 'reaction-rate', relation: 'AFFECTS', highlighted: false },
    { id: 'e4', source: 'temperature', target: 'reaction-rate', relation: 'AFFECTS', highlighted: false },
    { id: 'e5', source: 'temperature', target: 'denaturation', relation: 'CAUSES', highlighted: false },
    { id: 'e6', source: 'denaturation', target: 'active-site', relation: 'DESTROYS', highlighted: false },
    { id: 'e7', source: 'kinetics', target: 'enzyme', relation: 'STUDIES', highlighted: false },
    { id: 'e8', source: 'enzyme', target: 'catalyst', relation: 'IS_A', highlighted: false },
    { id: 'e9', source: 'inhibition', target: 'enzyme', relation: 'BLOCKS', highlighted: false },
    { id: 'e10', source: 'ph', target: 'enzyme', relation: 'AFFECTS', highlighted: false },
    { id: 'e11', source: 'enzyme', target: 'activation-energy', relation: 'LOWERS', highlighted: false },
    { id: 'e12', source: 'thermodynamics', target: 'kinetics', relation: 'PREREQUISITE', highlighted: false },
  ],
}

export const mockMessages: Message[] = [
  {
    id: '1',
    role: 'user',
    content: 'Why does the enzyme active site shape matter for reaction rate, and how does temperature affect this?',
    timestamp: new Date(Date.now() - 60000),
  },
  {
    id: '2',
    role: 'assistant',
    content:
      'The enzyme\'s active site has a specific 3D shape that is complementary to its substrate — this is the "lock and key" or induced-fit model. When the substrate fits precisely into the active site, the enzyme lowers the activation energy of the reaction, dramatically increasing the reaction rate.\n\nTemperature has a dual effect:\n\n1. **Up to the optimal temperature (~37°C for most human enzymes):** Increasing temperature raises the kinetic energy of molecules, so substrate-enzyme collisions are more frequent and forceful, increasing the reaction rate.\n\n2. **Above the optimal temperature:** Excessive heat breaks the hydrogen bonds and other non-covalent interactions that maintain the enzyme\'s 3D structure. This process — **denaturation** — distorts the active site shape. Once the active site no longer fits the substrate, catalytic activity drops sharply, often irreversibly.',
    routing_mode: 'graph',
    hops: ['enzyme', 'active-site', 'reaction-rate', 'temperature', 'denaturation'],
    reflection: {
      needs_retrieval: true,
      is_relevant: true,
      is_supported: true,
      is_useful: true,
      reasoning: 'Retrieved subgraph covers enzyme structure, active site binding, and temperature effects on denaturation — all directly relevant to the multi-hop question.',
    },
    evidence: [
      {
        id: 'c1',
        text: 'The active site of an enzyme is a region that has a specific shape, complementary to the substrate. This specificity allows the enzyme to selectively catalyze reactions.',
        source: 'Biochemistry Lecture 4 — Enzyme Structure',
        modality: 'slide',
        page: 12,
      },
      {
        id: 'c2',
        text: 'At temperatures above the optimum, the hydrogen bonds holding the enzyme\'s tertiary structure begin to break. This leads to denaturation, where the active site loses its shape.',
        source: 'Biochemistry Lecture 6 — Temperature & pH Effects',
        modality: 'video',
        timestamp: '14:32',
      },
      {
        id: 'c3',
        text: 'Reaction rate increases with temperature up to an optimum due to increased molecular collisions. Beyond this, enzyme denaturation causes a rapid decrease in rate.',
        source: 'Enzyme Kinetics — Textbook Ch. 8',
        modality: 'pdf',
        page: 203,
      },
    ],
    timestamp: new Date(Date.now() - 55000),
  },
]

export const highlightedGraph: GraphData = {
  nodes: [
    { id: 'enzyme', label: 'Enzyme', type: 'concept', highlighted: true },
    { id: 'active-site', label: 'Active Site', type: 'concept', highlighted: true },
    { id: 'substrate', label: 'Substrate', type: 'concept', highlighted: false },
    { id: 'reaction-rate', label: 'Reaction Rate', type: 'concept', highlighted: true },
    { id: 'temperature', label: 'Temperature', type: 'concept', highlighted: true },
    { id: 'kinetics', label: 'Enzyme Kinetics', type: 'topic', highlighted: false },
    { id: 'inhibition', label: 'Inhibition', type: 'concept', highlighted: false },
    { id: 'denaturation', label: 'Denaturation', type: 'concept', highlighted: true },
    { id: 'thermodynamics', label: 'Thermodynamics', type: 'topic', highlighted: false },
    { id: 'catalyst', label: 'Catalyst', type: 'concept', highlighted: false },
    { id: 'ph', label: 'pH Level', type: 'concept', highlighted: false },
    { id: 'activation-energy', label: 'Activation Energy', type: 'concept', highlighted: false },
  ],
  edges: [
    { id: 'e1', source: 'enzyme', target: 'active-site', relation: 'HAS_PART', highlighted: true },
    { id: 'e2', source: 'active-site', target: 'substrate', relation: 'BINDS', highlighted: false },
    { id: 'e3', source: 'enzyme', target: 'reaction-rate', relation: 'AFFECTS', highlighted: true },
    { id: 'e4', source: 'temperature', target: 'reaction-rate', relation: 'AFFECTS', highlighted: true },
    { id: 'e5', source: 'temperature', target: 'denaturation', relation: 'CAUSES', highlighted: true },
    { id: 'e6', source: 'denaturation', target: 'active-site', relation: 'DESTROYS', highlighted: true },
    { id: 'e7', source: 'kinetics', target: 'enzyme', relation: 'STUDIES', highlighted: false },
    { id: 'e8', source: 'enzyme', target: 'catalyst', relation: 'IS_A', highlighted: false },
    { id: 'e9', source: 'inhibition', target: 'enzyme', relation: 'BLOCKS', highlighted: false },
    { id: 'e10', source: 'ph', target: 'enzyme', relation: 'AFFECTS', highlighted: false },
    { id: 'e11', source: 'enzyme', target: 'activation-energy', relation: 'LOWERS', highlighted: false },
    { id: 'e12', source: 'thermodynamics', target: 'kinetics', relation: 'PREREQUISITE', highlighted: false },
  ],
}
