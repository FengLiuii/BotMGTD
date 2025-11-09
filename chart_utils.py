#!/usr/bin/env python3


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

def configure_chart_style():
    
    try:
        from font_config import configure_font
        configure_font()
        print("Chart style configured using font_config module")
    except ImportError:
        
        import platform
        if platform.system() == 'Linux' and os.path.exists('/mnt/c/Windows/Fonts/times.ttf'):
            try:
                import matplotlib.font_manager as fm
                fm.fontManager.addfont('/mnt/c/Windows/Fonts/times.ttf')
                font_family = 'Times New Roman'
            except:
                font_family = 'DejaVu Serif'
        else:
            font_family = 'Times New Roman'
        
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
        mpl.rcParams['axes.unicode_minus'] = False
        
        print(f"Chart style configured: {font_family} 16pt")

def save_chart(fig, filename, dpi=300, bbox_inches='tight'):
    
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, 
                facecolor='white', edgecolor='none')
    print(f"✅ Chart saved: {filename} (300dpi PNG)")

def create_attention_visualization(attention_info, output_file):
    """vis attention"""
    configure_chart_style()
    
    try:
        #
        fig = plt.figure(figsize=(18, 12))
        
        # first fusion
        first_fusion = attention_info['first_fusion']
        if first_fusion['importance_weights'] is not None:
            first_importance = np.asarray(first_fusion['importance_weights']).mean(axis=0)
            first_names = list(first_fusion['relation_names'])
            
            # sub1
            ax1 = fig.add_subplot(2, 3, 1)
            ax1.bar(first_names, first_importance, color='skyblue', width=0.6)
            ax1.set_title('First Fusion: Friend + Mention + Reply + Quote + Other')
            ax1.set_ylabel('Importance Weight')
            ax1.set_xticklabels(first_names, rotation=45, ha='right')
            
            # sub2
            if first_fusion['attention_weights'] is not None:
                ax2 = fig.add_subplot(2, 3, 2)
                data = np.asarray(first_fusion['attention_weights'][0])
                im = ax2.imshow(data, cmap='viridis')
                ax2.set_title('First Fusion Attention Patterns')
                ax2.set_xticks(np.arange(len(first_names)))
                ax2.set_yticks(np.arange(len(first_names)))
                ax2.set_xticklabels(first_names, rotation=45, ha='right')
                ax2.set_yticklabels(first_names)
                
                
                mean_val = float(np.mean(data))
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        ax2.text(j, i, f"{data[i, j]:.2f}",
                                 ha="center", va="center",
                                 color="white" if data[i, j] > mean_val else "black")
                cbar = fig.colorbar(im, ax=ax2)
                cbar.ax.tick_params(labelsize=16)
            
            # sub3
            ax3 = fig.add_subplot(2, 3, 3)
            ax3.pie(first_importance, labels=first_names, autopct='%1.1f%%',
                    textprops={'fontfamily': 'Times New Roman', 'fontsize': 16})
            ax3.set_title('First Fusion Importance Distribution')
        
        # second fusion
        second_fusion = attention_info['second_fusion']
        if second_fusion['importance_weights'] is not None:
            second_importance = np.asarray(second_fusion['importance_weights']).mean(axis=0)
            second_names = list(second_fusion['relation_names'])
            
            # sub4
            ax4 = fig.add_subplot(2, 3, 4)
            ax4.bar(second_names, second_importance, color='lightcoral', width=0.6)
            ax4.set_title('Second Fusion: Follower + First_Fused')
            ax4.set_ylabel('Importance Weight')
            ax4.set_xticklabels(second_names, rotation=45, ha='right')
            
            # sub5
            if second_fusion['attention_weights'] is not None:
                ax5 = fig.add_subplot(2, 3, 5)
                data2 = np.asarray(second_fusion['attention_weights'][0])
                im2 = ax5.imshow(data2, cmap='viridis')
                ax5.set_title('Second Fusion Attention Patterns')
                ax5.set_xticks(np.arange(len(second_names)))
                ax5.set_yticks(np.arange(len(second_names)))
                ax5.set_xticklabels(second_names, rotation=45, ha='right')
                ax5.set_yticklabels(second_names)
                
                mean_val2 = float(np.mean(data2))
                for i in range(data2.shape[0]):
                    for j in range(data2.shape[1]):
                        ax5.text(j, i, f"{data2[i, j]:.2f}",
                                 ha="center", va="center",
                                 color="white" if data2[i, j] > mean_val2 else "black")
                cbar2 = fig.colorbar(im2, ax=ax5)
                cbar2.ax.tick_params(labelsize=16)
            
            # sub6
            ax6 = fig.add_subplot(2, 3, 6)
            ax6.pie(second_importance, labels=second_names, autopct='%1.1f%%',
                    textprops={'fontfamily': 'Times New Roman', 'fontsize': 16})
            ax6.set_title('Second Fusion Importance Distribution')
        
        
        fig.tight_layout()
        save_chart(fig, output_file)
        plt.close(fig)
        
        return True
        
    except Exception as e:
        print(f" Failed to create attention visualization: {e}")
        return False

def create_architecture_diagram(output_file):
    
    configure_chart_style()
    
    try:
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        
        colors = {
            'input': '#E8F4FD',
            'process': '#B8E6B8', 
            'attention': '#FFE4B5',
            'output': '#FFB6C1',
            'target': '#DDA0DD'
        }
        
       
        ax.text(5, 7.5, 'Two-Stage Attention Fusion Architecture for MGTAB Dataset', 
                fontsize=16, fontweight='bold', ha='center')
        
        
        input_y = 6.5
        relation_names = ['Follower\n(Target)', 'Friend', 'Mention', 'Reply', 'Quote', 'Other']
        relation_colors = [colors['target']] + [colors['input']] * 5
        
        for i, (name, color) in enumerate(zip(relation_names, relation_colors)):
            x = 0.5 + i * 1.5
            from matplotlib.patches import FancyBboxPatch
            box = FancyBboxPatch((x-0.4, input_y-0.3), 0.8, 0.6, 
                                boxstyle="round,pad=0.05", 
                                facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(box)
            ax.text(x, input_y, name, ha='center', va='center', fontsize=10, fontweight='bold')
        
        
        first_fusion_y = 5
        ax.text(5, 5.3, 'First Stage: Attention Fusion', ha='center', fontsize=12, fontweight='bold')
        
        
        first_input_y = 4.7
        first_relations = ['Friend', 'Mention', 'Reply', 'Quote', 'Other']
        for i, name in enumerate(first_relations):
            x = 1.5 + i * 1.2
            from matplotlib.patches import FancyBboxPatch
            box = FancyBboxPatch((x-0.3, first_input_y-0.2), 0.6, 0.4, 
                                boxstyle="round,pad=0.03", 
                                facecolor=colors['input'], edgecolor='black', linewidth=1)
            ax.add_patch(box)
            ax.text(x, first_input_y, name, ha='center', va='center', fontsize=9)
        
        
        from matplotlib.patches import FancyBboxPatch
        attention1_box = FancyBboxPatch((4.2, 4.2), 1.6, 0.6, 
                                       boxstyle="round,pad=0.05", 
                                       facecolor=colors['attention'], edgecolor='black', linewidth=2)
        ax.add_patch(attention1_box)
        ax.text(5, 4.5, 'Multi-Head\nAttention', ha='center', va='center', fontsize=10, fontweight='bold')
        
        
        first_output_box = FancyBboxPatch((6.5, 4.2), 1.2, 0.6, 
                                         boxstyle="round,pad=0.05", 
                                         facecolor=colors['output'], edgecolor='black', linewidth=1.5)
        ax.add_patch(first_output_box)
        ax.text(7.1, 4.5, 'First\nFused\nVector', ha='center', va='center', fontsize=10, fontweight='bold')
        
       
        second_fusion_y = 3
        ax.text(5, 3.3, 'Second Stage: Follower + First Fused Vector', ha='center', fontsize=12, fontweight='bold')
        
        
        second_input_y = 2.7
        
        follower_box = FancyBboxPatch((2.5, second_input_y-0.2), 0.6, 0.4, 
                                     boxstyle="round,pad=0.03", 
                                     facecolor=colors['target'], edgecolor='black', linewidth=1)
        ax.add_patch(follower_box)
        ax.text(2.8, second_input_y, 'Follower', ha='center', va='center', fontsize=9)
        
        # First Fused Vector
        fused_box = FancyBboxPatch((4.5, second_input_y-0.2), 1.0, 0.4, 
                                  boxstyle="round,pad=0.03", 
                                  facecolor=colors['output'], edgecolor='black', linewidth=1)
        ax.add_patch(fused_box)
        ax.text(5, second_input_y, 'First Fused', ha='center', va='center', fontsize=9)
        
        
        attention2_box = FancyBboxPatch((3.8, 2.2), 1.6, 0.6, 
                                       boxstyle="round,pad=0.05", 
                                       facecolor=colors['attention'], edgecolor='black', linewidth=2)
        ax.add_patch(attention2_box)
        ax.text(4.6, 2.5, 'Multi-Head\nAttention', ha='center', va='center', fontsize=10, fontweight='bold')
        
       
        second_output_box = FancyBboxPatch((6.2, 2.2), 1.2, 0.6, 
                                          boxstyle="round,pad=0.05", 
                                          facecolor=colors['output'], edgecolor='black', linewidth=1.5)
        ax.add_patch(second_output_box)
        ax.text(6.8, 2.5, 'Final\nAuxiliary\nVector', ha='center', va='center', fontsize=10, fontweight='bold')
        
        
        target_y = 1.5
        target_box = FancyBboxPatch((1.5, target_y-0.3), 1.2, 0.6, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor=colors['target'], edgecolor='black', linewidth=2)
        ax.add_patch(target_box)
        ax.text(2.1, target_y, 'Target\nEmbedding\n(Follower)', ha='center', va='center', fontsize=10, fontweight='bold')
        
       
        final_fusion_box = FancyBboxPatch((4.5, 0.8), 2.0, 0.8, 
                                         boxstyle="round,pad=0.05", 
                                         facecolor=colors['process'], edgecolor='black', linewidth=2)
        ax.add_patch(final_fusion_box)
        ax.text(5.5, 1.2, 'Final Fusion\n(Concatenation + MLP)', ha='center', va='center', fontsize=11, fontweight='bold')
        
        
        final_output_box = FancyBboxPatch((7.5, 0.8), 1.2, 0.8, 
                                         boxstyle="round,pad=0.05", 
                                         facecolor=colors['output'], edgecolor='black', linewidth=2)
        ax.add_patch(final_output_box)
        ax.text(8.1, 1.2, 'Final\nEmbedding', ha='center', va='center', fontsize=11, fontweight='bold')
        
        
        plt.tight_layout()
        save_chart(fig, output_file)
        plt.close(fig)
        
        return True
        
    except Exception as e:
        print(f" Failed to create architecture diagram: {e}")
        return False

def create_comparison_chart(output_file):
    
    configure_chart_style()
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Relationship Aggregation Methods Comparison', fontsize=16, fontweight='bold')
        
        
        ax1.set_title('Current GRU Method (Problematic)', fontweight='bold', color='red')
        relations = ['Follow', 'Mention', 'Reply', 'Quote', 'Other']
        
        
        for i, rel in enumerate(relations):
            y_pos = 4 - i
            ax1.barh(y_pos, 1, height=0.6, color='lightcoral', alpha=0.7)
            ax1.text(0.5, y_pos, rel, ha='center', va='center', fontweight='bold')
            if i < len(relations) - 1:
                ax1.arrow(1.1, y_pos, 0.3, -1, head_width=0.1, head_length=0.1, fc='red', ec='red')
        
        ax1.set_xlim(0, 2)
        ax1.set_ylim(-0.5, 4.5)
        ax1.set_xlabel('Sequential Processing (Problematic)')
        ax1.set_ylabel('Relations')
        
        
        ax2.set_title('Improved Attention Method', fontweight='bold', color='green')
        ax2.bar(range(len(relations)), [0.3, 0.25, 0.2, 0.15, 0.1], 
                color='lightgreen', alpha=0.7)
        ax2.set_xticks(range(len(relations)))
        ax2.set_xticklabels(relations, rotation=45, ha='right')
        ax2.set_ylabel('Attention Weight')
        ax2.set_title('Parallel Processing with Attention Weights')
        
       
        methods = ['GRU', 'Attention', 'Sum', 'Concat']
        accuracy = [0.9118, 0.9122, 0.9008, 0.9056]
        colors = ['red', 'green', 'orange', 'blue']
        
        bars = ax3.bar(methods, accuracy, color=colors, alpha=0.7)
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Performance Comparison')
        ax3.set_ylim(0, 1)
        
       
        for bar, acc in zip(bars, accuracy):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.2f}', ha='center', va='bottom', fontweight='bold')
        
        
        complexity = [0, 0, 0, 0]  
        bars2 = ax4.bar(methods, complexity, color=colors, alpha=0.7)
        ax4.set_ylabel('Relative Computation Time')
        ax4.set_title('Computational Complexity')
        
       
        for bar, comp in zip(bars2, complexity):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{comp:.1f}x', ha='center', va='bottom', fontweight='bold')
        
        
        plt.tight_layout()
        save_chart(fig, output_file)
        plt.close(fig)
        
        return True
        
    except Exception as e:
        print(f" Failed to create comparison chart: {e}")
        return False

if __name__ == '__main__':
   
    print(" Testing chart generation utilities...")
    
    
    create_architecture_diagram('test_architecture_diagram.png')
    
    
    create_comparison_chart('test_comparison_chart.png')
    
    print(" Chart generation test completed!")
