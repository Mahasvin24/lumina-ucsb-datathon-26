import { QuestionRow } from "@/lib/dashboard-api";

type QuestionsTableProps = {
  questions: QuestionRow[];
  questionThresholdPct: number;
};

export function QuestionsTable({
  questions,
  questionThresholdPct,
}: QuestionsTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-left text-sm">
        <thead>
          <tr className="border-b border-zinc-200 text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
            <th className="px-2 py-2 font-medium">Question</th>
            <th className="px-2 py-2 font-medium">Class % Correct</th>
            <th className="px-2 py-2 font-medium">Correct / Attempts</th>
            <th className="px-2 py-2 font-medium">Skill Tags</th>
            <th className="px-2 py-2 font-medium">Flag</th>
          </tr>
        </thead>
        <tbody>
          {questions.map((question) => (
            <tr
              key={question.questionId}
              className="border-b border-zinc-100 text-zinc-800 dark:border-zinc-900 dark:text-zinc-200"
            >
              <td className="px-2 py-2">Q{question.questionId}</td>
              <td className="px-2 py-2">{question.classCorrectPct.toFixed(2)}%</td>
              <td className="px-2 py-2">
                {question.correctCount} / {question.attempts}
              </td>
              <td className="px-2 py-2">
                {question.skillTags.length > 0
                  ? question.skillTags.join(", ")
                  : "None"}
              </td>
              <td className="px-2 py-2">
                {question.classCorrectPct < questionThresholdPct ? "Flagged" : "OK"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
