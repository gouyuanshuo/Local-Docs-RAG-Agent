import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";

import { AskPanel } from "./AskPanel";

function renderPanel(error: string | null) {
  render(
    <AskPanel
      runtime=""
      question=""
      result={null}
      rawPayload="Waiting for a question..."
      isAsking={false}
      error={error}
      onRuntimeChange={() => undefined}
      onQuestionChange={() => undefined}
      onAsk={async () => undefined}
    />,
  );
}

it("shows ask errors without opening the raw payload", () => {
  renderPanel("Question must not be blank");

  expect(screen.getByRole("alert")).toHaveTextContent("Question must not be blank");
});

it("labels the question textarea", () => {
  renderPanel(null);

  expect(screen.getByRole("textbox", { name: "Question" })).toBeVisible();
});
