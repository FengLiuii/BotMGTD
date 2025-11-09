#!/usr/bin/env python3


import matplotlib as mpl
import matplotlib.pyplot as plt
import platform
import os

def configure_font():
    
    print(" Configuring matplotlib font...")
    
    
    if platform.system() == 'Windows':
        font_family = 'Times New Roman'
        print("   Platform: Windows - using Times New Roman")
    elif platform.system() == 'Linux':
        
        if os.path.exists('/mnt/c/Windows/Fonts/times.ttf'):
            try:
                import matplotlib.font_manager as fm
                fm.fontManager.addfont('/mnt/c/Windows/Fonts/times.ttf')
                font_family = 'Times New Roman'
                print("   Platform: WSL - using Windows Times New Roman")
            except:
                font_family = 'DejaVu Serif'
                print("   Platform: WSL - using DejaVu Serif (Times New Roman not available)")
        else:
            font_family = 'DejaVu Serif'
            print("   Platform: Linux - using DejaVu Serif")
    else:
        font_family = 'Times New Roman'
        print("   Platform: Other - using Times New Roman")
    
    # 设置字体
    mpl.rcParams['font.family'] = font_family
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['legend.fontsize'] = 16
    mpl.rcParams['figure.titlesize'] = 16
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    
   
    try:
       
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, 'Test', fontsize=16)
        plt.close(fig)
        
        print(f"   Font configured successfully!")
        print(f"   Font family: {mpl.rcParams['font.family']}")
        print(f"   Font size: {mpl.rcParams['font.size']}")
        
        
        current_font = mpl.font_manager.FontProperties(family=mpl.rcParams['font.family'])
        font_file = mpl.font_manager.findfont(current_font)
        print(f"   Actual font file: {os.path.basename(font_file)}")
        
        return True
        
    except Exception as e:
        print(f"  Font configuration warning: {e}")
        print("   Using default font...")
        return False

def get_available_fonts():
    
    fonts = [f.name for f in mpl.font_manager.fontManager.ttflist]
    return sorted(set(fonts))

def test_font_rendering():
    
    print(" Testing font rendering...")
    
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        
       
        x = np.arange(5)
        y = np.random.rand(5)
        labels = ['Friend', 'Mention', 'Reply', 'Quote', 'Other']
        
        
        bars = ax.bar(x, y, color='skyblue', alpha=0.7)
        
        
        ax.set_title('Font Rendering Test', fontsize=16, fontweight='bold')
        ax.set_xlabel('Relationship Types', fontsize=16)
        ax.set_ylabel('Importance Weight', fontsize=16)
        
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=16, rotation=45, ha='right')
        
        
        ax.tick_params(axis='y', labelsize=16)
        
        
        for bar, value in zip(bars, y):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=14)
        
        
        ax.legend(['Test Data'], fontsize=16, loc='upper right')
        
        
        plt.tight_layout()
        
        
        output_file = 'font_rendering_test.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Font rendering test completed!")
        print(f"   Test image saved as: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"  Font rendering test failed: {e}")
        return False

if __name__ == '__main__':
    import numpy as np
    
    
    success = configure_font()
    
    if success:
       
        test_font_rendering()
        
       
        fonts = get_available_fonts()
        print(f"\n Available fonts ({len(fonts)}):")
        for i, font in enumerate(fonts[:10]): 
            print(f"   {i+1}. {font}")
        if len(fonts) > 10:
            print(f"   ... and {len(fonts)-10} more")
    
    print("\n Font configuration completed!")
