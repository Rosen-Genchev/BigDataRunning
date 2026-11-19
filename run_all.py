import matplotlib.pyplot as plt

SHOW_PLOTS = False

if not SHOW_PLOTS:
    plt.show = lambda *args, **kwargs: None


def banner(text):
    print()
    print("#" * 60)
    print(text)
    print("#" * 60)


step = ""

try:
    step = "Stage 1 - generating the data"
    banner(step)
    import GenerateData

    step = "Stage 1 - loading and checking the data"
    banner(step)
    import LoadAndCheck

    step = "Stage 2 - preprocessing"
    banner(step)
    import CleanData

    step = "Stage 3 - statistics"
    banner(step)
    import Statistics

    step = "Stage 4 - time series and forecasting"
    banner(step)
    import TimeSeries

    step = "Stage 5 - customer reviews"
    banner(step)
    import Reviews

    step = "Stage 6 - segmentation and business analysis"
    banner(step)
    import Segmentation

except Exception as error:
    print()
    print("FAILED during:", step)
    print("Error:", error)
    raise

banner("ALL STAGES COMPLETED")
print("Now start the dashboard in the terminal:")
print("    streamlit run Dashboard.py")