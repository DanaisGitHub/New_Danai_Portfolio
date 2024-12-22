"use client";

import Image from "next/image";
import React from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import { BsArrowRight, BsLinkedin } from "react-icons/bs";
import { HiDownload } from "react-icons/hi";
import { FaGithubSquare } from "react-icons/fa";
import { useSectionInView } from "@/lib/hooks";
import { useActiveSectionContext } from "@/context/active-section-context";
import { Button as MovingBorder } from "@/components/ui/moving-border";
import { ProfilePic } from '@/lib/data'
import { Cover } from '@/components/ui/cover'
import { BackgroundGradient } from "@/components/ui/background-gradient";

export default function Intro() {
  const { ref } = useSectionInView("Home", 0.5);
  const { setActiveSection, setTimeOfLastClick } = useActiveSectionContext();

  return (
    <section
      ref={ref}
      id="home"
      className="max-lg:flex-col max-lg:block flex w-[100%] mb-28 text-center sm:mb-0 scroll-mt-[100rem] "
    >
      <div className="flex flex-wrap-reverse items-center py-7">
        <div className="  lg:w-1/2 items-center justify-center mx-auto flex">
          <div className="relative ">
            <motion.div
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{
                type: "tween",
                duration: 0.2,
              }}
            >
              <MovingBorder
                containerClassName="h-fit w-fit"
                borderClassName=" h-72 w-48 opacity-[0.8] bg-[radial-gradient(var(--red-600)_40%,transparent_60%)] "
                duration={2000}>
                <Image
                  src={ProfilePic}
                  alt="Danai Picture"
                  width="500"
                  height="500"
                  quality="95"
                  objectFit="fill"
                  priority={true}
                  loading="eager"
                  className=" rounded-3xl shadow-3xl -z-[1004] " />
              </MovingBorder>
            </motion.div>

            <motion.span
              className="absolute bottom-0 right-0 text-5xl"
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{
                type: "spring",
                stiffness: 150,
                delay: 0.2,
                duration: 0.9,
              }}
            >
              👋
            </motion.span>
          </div>
        </div>

        <div className="block lg:w-1/2 items-center justify-center max-lg:mb-7">
          <motion.h1
            className=" mt-[15vh] mb-10 items-center justify-center m-auto align-middle text-2xl font-medium !leading-[1.5] sm:text-4xl"
            initial={{ opacity: 0, y: 100 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <span className="font-bold">Hello, I'm  Danai Zerai.</span> I'm a{" "}
            <span className="font-bold">Software Developer</span>{" "}
            <span className="font-bold"></span> Aiming to be unreplaceable by {" "}
            <span className="font-bold">AI</span>. Or to become it{" "}
            <span className="underline">IT</span>.
          </motion.h1>
          <motion.div
            className="flex flex-col w-full sm:flex-row items-center justify-center gap-2 px-4 text-lg font-medium"
            initial={{ opacity: 0, y: 100 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              delay: 0.1,
            }}
          >

            <Link
              href="#contact"
              className="group bg-gray-900 text-white px-7 py-3 flex items-center gap-2 rounded-full outline-none focus:scale-110 hover:scale-110 hover:bg-gray-950 active:scale-105 transition"
              onClick={() => {
                setActiveSection("Contact");
                setTimeOfLastClick(Date.now());
              }}
            >
              Contact me here{" "}
              <BsArrowRight className="opacity-70 group-hover:translate-x-1 transition" />
            </Link>

            <Link
              className="group bg-white px-7 py-3 flex items-center gap-2 rounded-full outline-none focus:scale-110 hover:scale-110 active:scale-105 transition cursor-pointer borderBlack dark:bg-white/10"
              href="/CV.pdf"
              download
            >
              Download CV{" "}
              <HiDownload className="opacity-60 group-hover:translate-y-1 transition" />
            </Link>

            <Link
              className="bg-white p-4 text-gray-700 hover:text-gray-950 flex items-center gap-2 rounded-full focus:scale-[1.15] hover:scale-[1.15] active:scale-105 transition cursor-pointer borderBlack dark:bg-white/10 dark:text-white/60"
              href="https://www.linkedin.com/in/danai-zerai/"
              target="_blank"
            >
              <BsLinkedin />
            </Link>

            <Link
              className="bg-white p-4 text-gray-700 flex items-center gap-2 text-[1.35rem] rounded-full focus:scale-[1.15] hover:scale-[1.15] hover:text-gray-950 active:scale-105 transition cursor-pointer borderBlack dark:bg-white/10 dark:text-white/60"
              href="https://github.com/DanaisGitHub"
              target="_blank"
            >
              <FaGithubSquare />
            </Link>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
