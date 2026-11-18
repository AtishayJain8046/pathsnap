import "../../pathsnap/ui/app.css";

export const metadata = {
  title: "PathSnap Evaluator",
  description: "Compare live route predictions with the Folium geographic map.",
};

export default function RootLayout({children}) {
  return <html lang="en"><body>{children}</body></html>;
}
