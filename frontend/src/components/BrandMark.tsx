/** The shared original leaf-and-book asset is also the source of the website's small leaf icon. */
export default function BrandMark({ size = 'small', className = '' }: { size?: 'small' | 'medium' | 'large'; className?: string }) {
  return <span aria-hidden="true" className={`lumin-brand-mark lumin-brand-mark--${size} ${className}`}>
    <img src="/brand.svg" alt="" width="48" height="48" draggable={false} />
  </span>
}
