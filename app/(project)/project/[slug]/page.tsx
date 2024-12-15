import MaxWidthWrapper from '@/components/MaxWidthWrapper'
import React from 'react'
import { unified } from "unified"
import remarkParse from "remark-parse"
import remarkFrontmatter from "remark-frontmatter"
import remarkRehype from "remark-rehype"
import rehypeSlug from 'rehype-slug'
import rehypeStringify from "rehype-stringify"
import rehypeHighlight from "rehype-highlight"
import matter from "gray-matter"
import fs from "fs"
import Onthispage from '@/components/Onthispage'
import rehypeAutolinkHeadings from 'rehype-autolink-headings'
import { rehypePrettyCode } from 'rehype-pretty-code'
import { transformerCopyButton } from '@rehype-pretty/transformers'
import { Metadata, ResolvingMetadata } from 'next'
import { TracingBeam } from "@/components/ui/tracing-beam";
import { Cover } from '@/components/ui/cover'
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css'; // Import KaTeX CSS

import '../../../globals.css'
import './MdStyles.css'

type Props = {
    params: Promise<{ slug: string, title: string, description: string }>
    searchParams: Promise<{ [key: string]: string | string[] | undefined }> // defines anything as params
}

// https://ondrejsevcik.com/blog/building-perfect-markdown-processor-for-my-blog


export default async function BlogPage({ params }: { params: Promise<{ slug: string }> }) {
    const processor = unified()
        .use(remarkParse)
        .use(remarkRehype)
        .use(rehypeKatex) 
        .use(remarkMath) 
        .use(rehypeStringify)
        .use(rehypeSlug)
        .use(rehypePrettyCode, {
            theme: "github-dark",
            transformers: [
                transformerCopyButton({
                    visibility: 'always',
                    feedbackDuration: 3_000,
                }),
            ],
        },
        )
        .use(rehypeAutolinkHeadings)

    const param = await params
    const filePath = `content/${param.slug}/${param.slug}.md`
    const fileContent = fs.readFileSync(filePath, "utf-8");
    const { data, content } = matter(fileContent)
    const htmlContent = (await processor.process(content)).toString()
    return (
        <TracingBeam className="px-6">
            <MaxWidthWrapper className='dark:prose-invert pb-7 '>
                <div className='flex '>
                    <div className='max-lg:px-1 w-full '>
                        
                        <h1 className=" markdown-content text-center font-extrabold text-5xl max-sm:text-4xl " >
                            <Cover>{data.title}</Cover>
                        </h1>
                        {/* <Onthispage className="markdown-content max-lg:hidden " htmlContent={htmlContent} /> */}{/* Fix styling */}
                        <div className="markdown-content lg:text-lg md:text-md sm:text-sm " dangerouslySetInnerHTML={{ __html: htmlContent }}></div>
                        
                    </div>
                </div>
            </MaxWidthWrapper>
        </TracingBeam>

    )
}


export async function generateMetadata({ params, searchParams }: Props, parent: ResolvingMetadata): Promise<Metadata> {
    // read route params 
    const {slug} = await params
    const filePath = `content/${slug}/${slug}.md`
    const fileContent = fs.readFileSync(filePath, "utf-8");
    const { data } = matter(fileContent)
    return {
        title: `${data.title} - Danai's Project`,
        description: data.description
    }

}