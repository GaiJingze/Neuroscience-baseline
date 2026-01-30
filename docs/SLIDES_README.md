# Clustering Pipeline Presentation Slides

## 📁 Available Files

1. **`CLUSTERING_PIPELINE_SLIDES.md`** (Marp format)
   - Best for: VS Code + Marp extension
   - Features: Auto-generates PDF/HTML/PPTX

2. **`PRESENTATION_SLIDES.txt`** (Plain text)
   - Best for: Copy-paste to PowerPoint/Google Slides
   - Features: Simple, clean formatting

3. **`SLIDES_SIMPLE.md`** (Markdown format)
   - Best for: GitHub preview or simple markdown renderers
   - Features: Box-drawing characters for visual appeal

## 🎨 How to Use

### Option 1: Marp (Recommended for PDF/PPTX)

1. Install Marp:
   ```bash
   npm install -g @marp-team/marp-cli
   ```

2. Generate PDF:
   ```bash
   cd /hy-tmp/clustering
   marp CLUSTERING_PIPELINE_SLIDES.md -o presentation.pdf
   ```

3. Generate PowerPoint:
   ```bash
   marp CLUSTERING_PIPELINE_SLIDES.md -o presentation.pptx
   ```

### Option 2: VS Code + Marp Extension

1. Install "Marp for VS Code" extension
2. Open `CLUSTERING_PIPELINE_SLIDES.md`
3. Click "Export Slide Deck" in the sidebar
4. Choose PDF, PPTX, or HTML

### Option 3: Manual (PowerPoint/Keynote)

1. Open `PRESENTATION_SLIDES.txt`
2. Copy each slide section
3. Paste into PowerPoint/Keynote/Google Slides
4. Apply your preferred theme

## 📊 Slide Contents

### Slide 1: Title
- Project title and subtitle

### Slide 2: Pipeline Overview
- Framework description
- Workflow diagram
- Key features

### Slide 3: Datasets
- MNIST description
- Fashion-MNIST description
- Data preprocessing details

### Slide 4: Baselines
- FlyHash: Random projection baseline
- Krotov-Hopfield: Best performing method ⭐
- SoftHebb: Hebbian learning approach
- Diehl & Cook: STDP-based SNN

### Slide 5: Results
- Performance comparison table
- Key findings
- Conclusions

## 🎯 Customization Tips

### Update Results
If you have complete results, update the numbers in Slide 5:

```markdown
Method          NMI      ARI      ACC
────────────────────────────────────
Krotov          0.58±?   0.47±?   0.63±?
FlyHash         0.55±0.03 0.41±0.02 0.57±0.01
SoftHebb        0.18±0.00 0.09±0.00 0.21±0.00
```

### Add Figures
Consider adding:
- Pipeline architecture diagram
- Sample images from datasets
- Clustering visualization (t-SNE/UMAP)
- Performance comparison bar charts

### Adjust Length
- Current: 5 content slides + 1 title = 6 total
- To shorten: Combine Slides 3 & 4
- To expand: Split Slide 4 into 2 slides (2 baselines each)

## 🖼️ Suggested Visuals

1. **Slide 2**: Add pipeline flowchart
2. **Slide 3**: Show example images from both datasets
3. **Slide 4**: Add method diagrams or pseudocode
4. **Slide 5**: Include bar chart for performance comparison

## 📝 Notes

- All content is concise (≤7 bullet points per slide)
- Technical jargon is minimized
- Numbers are from actual experiments
- Easy to update with new results

## 🔄 Quick Updates

To update results after completing all experiments:

```bash
# Run this to get latest stats
cd /hy-tmp/clustering
python scripts/collect_all_results.py

# Then update the numbers in Slide 5
```

## 📧 Export Formats

Marp supports:
- PDF (best for printing)
- PPTX (editable in PowerPoint)
- HTML (interactive, web-friendly)
- PNG/JPEG (image sequence)
