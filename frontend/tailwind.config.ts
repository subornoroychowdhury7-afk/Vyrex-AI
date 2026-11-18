import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        vyrex: {
          50:  "#fff1f1",
          100: "#ffe0e0",
          200: "#ffc7c7",
          300: "#ffa0a0",
          400: "#ff6b6b",
          500: "#f83b3b",
          600: "#e51a1a",
          700: "#c11111",
          800: "#a01111",
          900: "#841616",
          950: "#480707",
        },
      },
      animation: {
        "fade-in": "fadeIn 0.25s ease-out",
        "slide-up": "slideUp 0.3s ease-out",
      },
      keyframes: {
        fadeIn: { from: { opacity: "0" }, to: { opacity: "1" } },
        slideUp: { from: { opacity: "0", transform: "translateY(10px)" }, to: { opacity: "1", transform: "translateY(0)" } },
      },
    },
  },
  plugins: [],
};
export default config;
