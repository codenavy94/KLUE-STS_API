from flask import Flask, render_template
import pandas as pd


def create_app():
    app = Flask(__name__)
    
    from project.view.main import main_bp

    app.register_blueprint(main_bp)
    
    return app
    
if __name__ == "__main__":
    app.run(debug=True)