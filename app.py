"""
=========================================================
app.py

Flask Backend for:
AI-Based Mechanical CAD Image Difference Detection System
=========================================================
"""

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from pathlib import Path

# Import custom modules
from compare import compare_images
from utils import allowed_file

# -------------------------------------------------------
# FLASK APP CONFIGURATION
# -------------------------------------------------------

app = Flask(__name__)
app.secret_key = "cad_difference_secret_key"

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


# -------------------------------------------------------
# HOME ROUTE
# -------------------------------------------------------

@app.route("/")
def index():
    """
    Render upload page.
    """
    return render_template("index.html")


# -------------------------------------------------------
# FILE UPLOAD + PROCESS ROUTE
# -------------------------------------------------------

@app.route("/compare", methods=["POST"])
def compare():
    """
    Handles image upload and triggers comparison pipeline.
    """

    try:
        # Check files
        if "image_a" not in request.files or "image_b" not in request.files:
            flash("Both images are required!", "danger")
            return redirect(url_for("index"))

        file_a = request.files["image_a"]
        file_b = request.files["image_b"]

        # Validate filenames
        if file_a.filename == "" or file_b.filename == "":
            flash("No file selected!", "danger")
            return redirect(url_for("index"))

        # Validate file types
        if not (allowed_file(file_a.filename) and allowed_file(file_b.filename)):
            flash("Invalid file format! Use PNG, JPG, JPEG.", "danger")
            return redirect(url_for("index"))

        # Secure filenames
        filename_a = secure_filename(file_a.filename)
        filename_b = secure_filename(file_b.filename)

        path_a = UPLOAD_FOLDER / filename_a
        path_b = UPLOAD_FOLDER / filename_b

        # Save uploaded files
        file_a.save(path_a)
        file_b.save(path_b)

        # ---------------------------------------------------
        # CALL CORE CV PIPELINE
        # ---------------------------------------------------

        result = compare_images(str(path_a), str(path_b))

        # ---------------------------------------------------
        # RENDER RESULT PAGE
        # ---------------------------------------------------

        return render_template(
            "result.html",
            original_a=url_for("static", filename=f"uploads/{filename_a}"),
            original_b=url_for("static", filename=f"uploads/{filename_b}"),
            aligned_image=url_for("static", filename=f"results/{Path(result['aligned_image']).name}"),
            difference_mask=url_for("static", filename=f"results/{Path(result['difference_mask']).name}"),
            highlighted_image=url_for("static", filename=f"results/{Path(result['highlighted_image']).name}"),

            ssim_score=result["ssim_score"],
            changed_regions=result["changed_regions"],
            changed_pixels=result["changed_pixels"],
            percentage_changed=result["percentage_changed"],
            summary=result["summary"]
        )

    except Exception as e:
        flash(f"Error processing images: {str(e)}", "danger")
        return redirect(url_for("index"))


# -------------------------------------------------------
# ERROR HANDLING ROUTE (OPTIONAL)
# -------------------------------------------------------

@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404 - Page Not Found</h1>", 404


@app.errorhandler(500)
def internal_error(e):
    return "<h1>500 - Internal Server Error</h1>", 500


# -------------------------------------------------------
# RUN SERVER
# -------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)