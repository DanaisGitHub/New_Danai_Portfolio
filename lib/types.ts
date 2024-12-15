import { links } from "./data";

export type SectionName = (typeof links)[number]["name"];
export type ProjectPrevType = {
    slug: string;
    title: string;
    description: string;
    imageUrl?: any;
    labels?: Array<string>
    github?: string
  
  };
