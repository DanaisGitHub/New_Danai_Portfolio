import React from 'react'
import { cn } from '@/lib/utils'

type MaxWidthProps = {
    className: string
    children: React.ReactNode
}

const MaxWidthWrapper = ({className, children}:MaxWidthProps) => {
  return (
    <div className={cn('mx-auto max-w-screen-2xl w-full', className)}>
        {children}
    </div>
  )
}

export default MaxWidthWrapper