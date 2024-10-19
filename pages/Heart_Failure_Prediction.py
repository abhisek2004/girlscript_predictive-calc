from page_handler import PageHandler

# Initialize the PageHandler with your pages configuration
page_handler = PageHandler("pages/pages.json")

# Automatically render the Heart Failure Prediction app
page_handler.render_page("Heart Failure Prediction")
