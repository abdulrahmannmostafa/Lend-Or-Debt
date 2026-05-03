import { globalStyles } from "./styles";
import { UploadSection } from "./components/UploadSection";
import { RunPhase } from "./components/RunPhase";
import { EDASection } from "./components/EDASection";

export default function App() {
  return (
    <>
      <style>{globalStyles}</style>

      <header className="top-bar">
        <h1>Lend-or-Debt — ML Pipeline</h1>
      </header>

      <div className="grid">
        <UploadSection />

        <RunPhase
          title="Data Cleaning"
          icon="🧹"
          endpoint="cleaning"
          desc="Runs src/pipeline/data_cleaning.py __main__ inside the project root."
        />

        <RunPhase
          title="Data Transformation"
          icon="⚙️"
          endpoint="transformation"
          desc="Runs src/pipeline/data_transformation.py __main__ inside the project root."
        />

        <EDASection />
      </div>
    </>
  );
}
