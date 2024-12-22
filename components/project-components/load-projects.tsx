import React from "react";
import { Button, buttonVariants } from "@/components/ui/button";
import fs, { readFileSync, readFile } from "fs";
import matter from "gray-matter";
import Link from "next/link";
import { Metadata } from "next";
import ProjectCard from '@/components/project-components/project-card'


type ProjectPrevType = {
  slug: string;
  title: string;
  description: string;
  imageUrl?: string;
  labels?: Array<string>
  github?: string

};

// reads all files in content
const dirContent = fs.readdirSync("content", { encoding: "utf-8", withFileTypes: false, recursive: false })

// reads all file 
const projects: ProjectPrevType[] = dirContent.map(file => {

  const fileContent = readFileSync(`content/${file}/${file}.md`, { encoding: "utf-8" }); //ASYNC is buggy may need a loading screen on app to
  const { data } = matter(fileContent)
  const value: ProjectPrevType = {
    slug: data.slug,
    title: data.title,
    description: data.description,
    imageUrl: data?.imageUrl, //TODO: solve
    labels: data?.labels,
    github: data?.github
  }
  return value
})

console.log(projects)




export default function BlogList() {
  return (
    <div className="flex flex-col min-w-full ">
      {projects.map((project, index) => (
        <React.Fragment key={index}>
          <ProjectCard {...project} />
        </React.Fragment>
      ))}
    </div>
  );
};









