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
    <html lang="en" className="!scroll-smooth dark ">

      <body
        className={`${inter.className}  relative pt-28 sm:pt-36 bg-gray-900 text-gray-50 text-opacity-90 `}
      >
        <StarsBackground className="-z-50" starDensity={0.00045} twinkleProbability={0.5} allStarsTwinkle={false} />
        {children}
        <Footer />
        <Toaster position="top-right" />

      </body>
    </html>
  );
}
