"use client";

import React, { useEffect } from "react";
import SectionHeading from "../section-heading";
import { useSectionInView } from "@/lib/hooks";
import { useActiveSectionContext } from "@/context/active-section-context";
import { useRef } from "react";





export default function Projects({ children }: any) {
    const { ref } = useSectionInView("Projects", 0.5);
    const refs = useRef<HTMLDivElement>(null);
    const { setActiveSection, setTimeOfLastClick } = useActiveSectionContext();

    return (
        <section ref={ref} id="projects" className="scroll-mt-28 mb-28 ">
            <SectionHeading>My projects</SectionHeading>
            <div ref= {refs} className="flex">
                {children}
            </div>
        </section>
    );
}


