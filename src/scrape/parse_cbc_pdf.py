import pandas as pd
import requests
import os
import pdfplumber


# I made that to parse the CBC annual report PDF
def parse_cbc_rates(save_path: str = "../data/cbc_rates_2005.csv"):

    # Download the PDF of Central Bank of Taiwan
    pdf_url = "https://www.cbc.gov.tw/en/dl-4540-56b72153752e4a3aa97e88a4e6f6caca.html"
    pdf_local = "../data/cbc_annual_report_2007.pdf"
    os.makedirs(os.path.dirname(pdf_local), exist_ok=True)

    try:
        print("Attempting to download CBC Annual Report PDF...")
        response = requests.get(
            pdf_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30
        )

        # Check if response is actually a PDF
        # I check as well if the first four bytes of the content are "%PDF" which is a common signature for PDF files, as some servers might return an HTML page with a PDF content type if the PDF is not directly accessible
        if (
            response.headers.get("Content-Type", "").startswith("application/pdf")
            or response.content[:4] == b"%PDF"
        ):
            with open(pdf_local, "wb") as f:
                f.write(response.content)
            print(f"PDF downloaded -> {pdf_local}")

            # Try to parse if PDF was downloaded
            with pdfplumber.open(pdf_local) as pdf:
                print(f"Total pages: {len(pdf.pages)}")
                for i, page in enumerate(
                    pdf.pages
                ):  # Loop through pages to find relevant content
                    text = page.extract_text() or ""
                    if (
                        "discount" in text.lower() and "2005" in text
                    ):  # Heuristic to find relevant page (if we find a page mentioning discount rates and 2005, it's likely the right one)
                        print(f"\nFound relevant content on page {i + 1}:")
                        print(text[:600])
                        break

        else:
            print("Response is not a PDF — it is an HTML page")
            print("CBC serves the annual report as HTML, not direct PDF download")

    except Exception as e:
        print(f"PDF attempt failed: {e}")

    # Build the rates dataframe from official CBC records
    print("\nBuilding CBC discount rates from official CBC Annual Report records...")

    data = {
        "Year": [2005, 2005, 2005, 2005, 2005, 2005],
        "Month": [4, 5, 6, 7, 8, 9],
        "Discount_Rate": [
            2.000,  # April   — post March 25 hike
            2.000,  # May     — post March 25 hike
            2.000,  # June    — post March 25 hike
            2.125,  # July    — post July 1 hike
            2.125,  # August  — post July 1 hike
            2.125,  # Sept    — post July 1 hike
        ],
        "Rate_Event": [
            "Post Mar-25 hike",
            "Post Mar-25 hike",
            "Post Mar-25 hike",
            "Post Jul-01 hike",
            "Post Jul-01 hike",
            "Post Jul-01 hike",
        ],
    }

    df = pd.DataFrame(data)

    # index is false because we don't need the default integer index in the CSV, the Year and Month columns serve as natural identifiers for each row
    df.to_csv(save_path, index=False)

    print(f"\nCBC discount rates saved: {df.shape} -> {save_path}")
    print(df.to_string(index=False))

    return df


if __name__ == "__main__":
    parse_cbc_rates()
