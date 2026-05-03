import { globalStyles } from "./styles";
import { UploadSection } from "./components/UploadSection";
import { RunPhase } from "./components/RunPhase";
import { EDASection } from "./components/EDASection";
import Intro from "./components/Intro";

export default function App() {
  return (
    <>
      <style>{globalStyles}</style>

      <header className="top-bar">
        <h1>Lend-or-Debt — ML Pipeline</h1>
      </header>
     
      < Intro />

      <div className="grid">
    
        <UploadSection />

        <RunPhase
          title="Data Cleaning"
          icon="🧹"
          endpoint="cleaning"
          desc="Runs Cleaning Phase."
        />

        <RunPhase
          title="Data Transformation"
          icon="⚙️"
          endpoint="transformation"
          desc="Runs Transformation Phase."
        />

        <EDASection />
      </div>
    </>
  );
}
