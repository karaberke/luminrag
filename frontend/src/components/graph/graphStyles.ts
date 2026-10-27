import { DEFAULT_STYLES } from 'sigma'
import type { SigmaOptions } from 'sigma/types'
import { graphPalette } from '../../utils/appearance'

const fullLabel = (attrs: Record<string, unknown>) => String(attrs.name ?? attrs.label ?? '')
const overviewLabel = (attrs: Record<string, unknown>) => {
  const label = fullLabel(attrs)
  return label.length > 28 ? `${label.slice(0, 27)}…` : label
}

export function graphOptions(dark: boolean): Partial<SigmaOptions> {
  const colors = graphPalette(dark)
  return {
    settings: { enableNodeDrag: true, autoRescale: 'once', enableCameraRotation: false, itemSizesReference: 'screen', zoomToSizeRatioFunction: (ratio) => ratio, allowInvalidContainer: true,
      enableEdgeEvents: true, nodeLabelEvents: 'extend', labelDensity: 0.3, labelGridCellSize: 200 },
    styles: {
      nodes: [DEFAULT_STYLES.nodes, { color: (attrs) => colors[attrs.node_type] ?? colors.content,
        label: overviewLabel, labelColor: colors.label, labelSize: 12, labelFont: 'Nunito', labelPosition: 'below',
        labelBackgroundColor: colors.backdrop, labelBackgroundPadding: 2, backdropColor: colors.backdrop,
        shape: (attrs) => attrs.node_type === 'note' ? 'square' : 'circle',
        opacity: (attrs) => attrs.dimmed ? 0.15 : 1,
      }, { whenState: 'isHighlighted', then: { color: colors.path, backdropVisibility: 'visible', backdropArea: 'node', backdropColor: colors.backdrop, backdropBorderColor: colors.path, backdropBorderWidth: 2 } },
        { whenData: { selected: true }, then: { label: fullLabel, labelVisibility: 'visible', backdropVisibility: 'visible', backdropArea: 'both', backdropColor: colors.backdrop, backdropBorderColor: colors.note, backdropBorderWidth: 2 } },
        { whenState: 'isHovered', then: { label: fullLabel, labelVisibility: 'visible', opacity: 1, backdropVisibility: 'visible', backdropArea: 'both', backdropColor: colors.backdrop } }],
      edges: [DEFAULT_STYLES.edges, { color: colors.edge, opacity: (attrs) => attrs.dimmed ? 0.06 : 0.35,
        labelVisibility: 'hidden', labelColor: colors.label, labelSize: 10, labelFont: 'Nunito', head: (attrs) => attrs.relation === 'RELATED_TO' ? 'none' : 'arrow' },
        { whenState: 'isHighlighted', then: { color: colors.path, size: 1.8, opacity: 0.85 } },
        { whenState: 'isHovered', then: { opacity: 1, size: 2, labelVisibility: (attrs) => attrs.relation === 'RELATED_TO' ? 'hidden' : 'visible', labelBackgroundColor: colors.backdrop, labelBackgroundPadding: 3 } }],
      stage: { background: colors.stage },
    },
  }
}
