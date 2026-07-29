from __future__ import annotations

import gradio as gr

import predict


def run_prediction(lyrics: str, country: str):
    if not lyrics or not lyrics.strip():
        return "Enter some lyrics first.", {}, "", ""

    result = predict.predict(lyrics, country)

    score_text = f"Predicted score: {result['predicted_score']:.1f}"

    place_text = (
        f"Estimated place: ~{result['estimated_place']} out of 26\n"
        f"(approximate — based on how this score compares to recent years' "
        f"historical results, not a simulated contest against real competitors)"
    )

    breakdown_text = (
        f"Contribution breakdown (linear model, exact):\n"
        f"  Lyrics/theme contribution: {result['theme_contribution']:+.1f}\n"
        f"  Country contribution: {result['country_contribution']:+.1f}\n"
        f"  Baseline (intercept): {result['intercept']:+.1f}"
    )

    return score_text, result["theme_scores"], place_text, breakdown_text


def build_app() -> gr.Blocks:
    countries = predict.known_countries()

    with gr.Blocks(title="Eurovision Score Predictor") as demo:
        gr.Markdown(
            "# Eurovision Score Predictor\n"
            "Type in lyrics for a new song and pick the country sending it, "
            "to get a predicted contest score and estimated place."
        )
        with gr.Row():
            lyrics_box = gr.Textbox(
                label="Song lyrics", lines=10, placeholder="Paste or write lyrics here..."
            )
            country_dropdown = gr.Dropdown(
                choices=countries, label="Country", value=countries[0] if countries else None
            )

        submit_btn = gr.Button("Predict", variant="primary")

        score_output = gr.Textbox(label="Score")
        theme_output = gr.Label(label="Theme scores")
        place_output = gr.Textbox(label="Estimated place")
        breakdown_output = gr.Textbox(label="Breakdown")

        submit_btn.click(
            fn=run_prediction,
            inputs=[lyrics_box, country_dropdown],
            outputs=[score_output, theme_output, place_output, breakdown_output],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
