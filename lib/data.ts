import React from "react";
import { CgWorkAlt } from "react-icons/cg";
import { FaReact } from "react-icons/fa";
import { LuGraduationCap } from "react-icons/lu";
import corpcommentImg from "@/public/corpcomment.png";
import rmtdevImg from "@/public/rmtdev.png";
import wordanalyticsImg from "@/public/wordanalytics.png";
import ImportedProfilePicture from '@/public/profile-pic.webp'
import previewImgImport from '@/public/preview.webp'


export const ProfilePic = ImportedProfilePicture



export const aboutMeText = `Lets keep it short🩳, I've been a developer since I can remember, engineering everything from web-apps to robot controllers. I am weirdly passionate about software, especially it’s problems, encountering problems and solving them is a drug💊 I cannot resist. With the advent of AI and greater software demand I aspire to be a true full stack developer, a generalist in my field. I hope that this is portrayed in my project portfolio 

In my non-coding time, I gym🏋🏾, the end.  `

export const links = [{
  name: "Home",
  hash: "#home",
},
{
  name: "About",
  hash: "#about",
},
{
  name: "Projects",
  hash: "#projects",
},
{
  name: "Skills",
  hash: "#skills",
},
{
  name: "Experience",
  hash: "#experience",
},
{
  name: "Contact",
  hash: "#contact",
},
] as const;

export const experiencesData = [
  {
    title: "Power Platform Developer                                                                                               ",
    location: "Manchester Uk                                                                                               ",
    description:
      "A power platform and software developer, engineering a plethora of solutions within the Microsoft eco-space.                                                                                               ",
    icon: React.createElement(CgWorkAlt),
    date: "Sept 2024 - Present",
  },
  {
    title: "Computer Science BSC, First Class                                                                                               ",
    location: "University Of Liverpool, Liverpool, UK",
    description:
      "",
    icon: React.createElement(LuGraduationCap),
    date: "Sept 2021 - July 2024",
  },
  {
    title: "Sales Agent", 
    location: "Carole Nash, Manchester, UK                                                                                                    ",
    description:
      ` A sales role which had strict KPI's to achieve and a high stress environment to achieve them in`,
    icon: React.createElement(CgWorkAlt),
    date: "Feb 2021 - Aug 2021",
  },
] as const;


export const skillsData = [
  "HTML / CSS / Tailwind",
  "JavaScript / TypeScript",
  "C# / .NET",
  "React / NEXT.js",
  "Node.js",
  "Git",
  "Python",
  "PyTorch",
  "Dart",
  "Flask",
  "Flutter",
  "Keras",
  "Java",
  "WeBots",
  "TensorFlow",
  "Framer Motion",
] as const;


