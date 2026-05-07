import pandas as pd
import matplotlib.pyplot as plt
import os

data_path = r"e:\Stock prediction\data\processed\merged\NABIL_train.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    # clean columns
    df.columns = df.columns.str.strip().str.lower()
    
    if 'close' in df.columns:
        y = 'close'
    elif 'ltp' in df.columns:
        y = 'ltp'
    elif 'target' in df.columns:
        y = 'target'
    else:
        y = df.columns[1]
    
    # take last 100 days for better visualization
    df = df.tail(100).reset_index(drop=True)
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    
    ax.plot(df.index, df[y], color='#38bdf8', linewidth=2.5, label='NABIL LTP')
    
    # Optional 20 day moving average
    ma20 = df[y].rolling(10).mean()
    ax.plot(df.index, ma20, color='#c084fc', linewidth=1.5, linestyle='--', label='10-Day MA')
    
    # Fill under curve
    ax.fill_between(df.index, df[y], alpha=0.1, color='#38bdf8')

    # Hide spines
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
        
    ax.tick_params(colors='#64748b', bottom=False, left=False)
    ax.grid(color='#1e293b', linestyle='--', linewidth=0.5)
    
    ax.set_title("Nabil Bank Limited (NABIL) - Last 100 Trading Days", color='#f8fafc', pad=20, fontsize=14, fontweight='bold', loc='left')
    plt.legend(frameon=False, loc='upper left', labelcolor='#e2e8f0')
    
    plt.tight_layout()
    plt.savefig(r"e:\Stock prediction\static\hero.png", dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=True)
    print("Saved plot to hero.png")
else:
    print("Data not found")
