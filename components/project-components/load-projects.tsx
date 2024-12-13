import React from "react";
import { Button, buttonVariants } from "@/components/ui/button";
import fs, { readFileSync } from "fs";
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

};

// reads all files in content
const dirContent = fs.readdirSync("content", "utf-8")

// reads all file 
const projects: ProjectPrevType[] = dirContent.map(file => {
  const fileContent = readFileSync(`content/${file}`, "utf-8"); //TURN ASYNC
  const { data } = matter(fileContent)
  const value: ProjectPrevType = {
    slug: data.slug,
    title: data.title,
    description: data.description,
    imageUrl: data?.imageUrl,
    labels: data?.labels
  }
  return value
})

console.log(projects)




const BlogList = () => {
  return (
    <div className="flex flex-wrap justify-around min-w-full ">
      {projects.map((project, index) => (
        <React.Fragment key={index}>
          <ProjectCard {...project} />
        </React.Fragment>
      ))}
    </div>
  );
};



export default BlogList;





