import Header from "@/components/header";
import "../globals.css";
import { Inter } from "next/font/google";
import ActiveSectionContextProvider from "@/context/active-section-context";
import Footer from "@/components/footer";
//import ThemeSwitch from "@/components/theme-switch";
import ThemeContextProvider from "@/context/theme-context";
import { Toaster } from "react-hot-toast";
import { ShootingStars } from "@/components/ui/shooting-stars";
import { StarsBackground } from "@/components/ui/stars-background";


const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "Danai Zerai | Portfolio",
  description: "Danai is self-proclaimed model who happens to be a full-stack developer enphasising his work recently on the backend",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className=" !scroll-smooth dark ">

      <body
        className={`${inter.className}  relative pt-28 sm:pt-36 bg-gray-900 text-gray-50 text-opacity-90 `}
      >
        <StarsBackground className="-z-50" starDensity={0.00065} twinkleProbability={0.7} allStarsTwinkle={false}/>
        <ShootingStars className="-z-50" minDelay={500} maxDelay={2000}/>


        <div className=" absolute top-[-6rem] right-[11rem] max-w-screen-xl -z-[51] h-[31.25rem] w-[31.25rem] rounded-full blur-[10rem] sm:w-[68.75rem] bg-[#94626382]"></div>
        <div className=" absolute top-[-1rem] left-[-35rem] max-w-screen-xl -z-[51] h-[31.25rem] w-[50rem] rounded-full blur-[10rem] sm:w-[68.75rem] md:left-[-33rem] lg:left-[-28rem] xl:left-[-15rem] 2xl:left-[-5rem] bg-[#676394aa]"></div>
        
        <ThemeContextProvider>
          <ActiveSectionContextProvider >

            <Header />
            {children}
            <Footer />
            <Toaster position="top-right" />
          </ActiveSectionContextProvider>
        </ThemeContextProvider>




      </body>
    </html>
  );
}
