import { lazy, Suspense } from 'react'
import type { RichTextProps } from './RichText'

// Download the markdown/math renderer only when there is content to display.
const Renderer = lazy(() => import('./RichText').then(({ RichText }) => ({ default: RichText })))

export default function LazyRichText(props: RichTextProps) {
  return (
    <Suspense fallback={<div className={props.className}>{props.text}</div>}>
      <Renderer {...props} />
    </Suspense>
  )
}
