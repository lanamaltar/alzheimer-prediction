import { Component, Input } from '@angular/core';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';
import { RouterLink } from '@angular/router';
import { CommonModule, PercentPipe } from '@angular/common';

@Component({
  selector: 'app-image-predict-result',
  standalone: true,
  imports: [CommonModule, PercentPipe, RouterLink],
  templateUrl: './image-predict-result.component.html',
  styleUrls: ['./image-predict-result.component.scss']
})
export class ImagePredictResultComponent {
  private backendBase = 'http://localhost:5000';

  private normalizeUrl(u?: string | null): string | null {
    if (!u) return null;
    return u.startsWith('http') ? u : this.backendBase + u;
  }
  getTop2Summary(): { name: string, prob: number }[] {
    if (!this.probabilities || !this.classNames) return [];
    // Pair class names and probabilities, sort descending
    const pairs = this.classNames.map((name: string, i: number) => ({ name, prob: this.probabilities![i] * 100 }));
    pairs.sort((a: any, b: any) => b.prob - a.prob);
    return pairs.slice(0, 2);
  }

  getPlainExplanation(): string {
    if (!this.probabilities || !this.classNames) return '';
    const pairs = this.classNames.map((name: string, i: number) => ({ name, prob: this.probabilities![i] * 100 }));
    pairs.sort((a: any, b: any) => b.prob - a.prob);
    const top = pairs[0];
    const second = pairs[1];
    if (!top) return '';
    if (top.prob > 70) {
      return `The model was confident: “${top.name}” (${top.prob.toFixed(2)}%).`;
    } else if (second) {
      return `The model leaned toward “${top.name}” (${top.prob.toFixed(2)}%), but also considered “${second.name}” (${second.prob.toFixed(2)}%). This means the result is uncertain.`;
    } else {
      return `The model was uncertain, with probabilities spread across classes.`;
    }
  }

  // Rule-based clinical interpretation using combined dementia probability
  getDementiaRuleExplanation(): string {
    if (!this.probabilities || !this.classNames) return '';
    const nonDemented = this.getStageProbability('Non Demented'); // percent
    const dementiaSum = this.getDementiaCombinedProbability(); // percent

    if (nonDemented >= 80) {
      return `Low risk: The model strongly favors “Non Demented” (${nonDemented.toFixed(2)}%).`;
    }
    if (dementiaSum >= 60) {
      return `High likelihood of early Alzheimer’s: The model shows strong indications of an early stage of Alzheimer’s disease.`;
    }
    if (dementiaSum >= 20) {
  return `Possible early Alzheimer’s: The model shows moderate indications of an early stage of Alzheimer’s disease.`;
    }
    return `Inconclusive: Results do not meet clear thresholds.`;
  }

  // Sum probabilities for dementia-positive stages (Very Mild, Mild, Moderate)
  getDementiaCombinedProbability(): number {
    const stages = ['Very Mild Dementia', 'Mild Dementia', 'Moderate Dementia'];
    return stages.reduce((acc, s) => acc + this.getStageProbability(s), 0);
  }
  @Input() result: string | null = null;
  @Input() confidence: number | null = null;
  @Input() probabilities: number[] | null = null;
  @Input() classNames: string[] | null = null;
  @Input() entropy: number | null = null;
  @Input() reliability: string | null = null;
  @Input() timestamp: Date | string | null = null;
  @Input() predictionId: string | null = null;
  @Input() modelInfo: any = null;
  @Input() inputDetails: any = null;
  @Input() gradCamUrl: string | null = null;
  @Input() downloadLinks: any = null;
  @Input() disclaimer: string | null = null;

  async printResultCardAsPDF() {
    const card = document.getElementById('result-card') as HTMLElement;
    if (!card) {
      alert('Result card not found.');
      return;
    }

    // Target only the button and link for exclusion
    const footerButton = document.querySelector('#result-footer button') as HTMLElement;
    const footerLink = document.querySelector('#result-footer a') as HTMLElement;

    const originalButtonDisplay = footerButton?.style.display;
    const originalLinkDisplay = footerLink?.style.display;

    if (footerButton) footerButton.style.display = 'none';
    if (footerLink) footerLink.style.display = 'none';

    // Use html2canvas to capture the result card, excluding the hidden elements
    const canvas = await html2canvas(card, {
      scale: 2,
      useCORS: true, // Ensure cross-origin images are included
    });

    // Restore the original display of the button and link
    if (footerButton) footerButton.style.display = originalButtonDisplay || '';
    if (footerLink) footerLink.style.display = originalLinkDisplay || '';

    const imgData = canvas.toDataURL('image/png');
    const jsPdfInstance = new jsPDF({
      orientation: canvas.width > canvas.height ? 'landscape' : 'portrait',
      unit: 'px',
      format: [canvas.width, canvas.height],
    });

    jsPdfInstance.addImage(imgData, 'PNG', 0, 0, canvas.width, canvas.height);
    jsPdfInstance.save('prediction-result-card.pdf');
  }

  // Text-preserving export via browser's Print to PDF (same behavior as numeric result card)
  printResultCardViaBrowser() {
    const card = document.querySelector('#result-card') as HTMLElement | null;
    if (!card) {
      alert('Result card not found.');
      return;
    }

    // Clone the card (images are preserved; canvases would be replaced if present)
    const clone = card.cloneNode(true) as HTMLElement;

    // Replace canvases in clone with images copied from original canvases (no-op if none)
    const origCanvases = Array.from(card.querySelectorAll('canvas')) as HTMLCanvasElement[];
    const cloneCanvases = Array.from(clone.querySelectorAll('canvas')) as HTMLCanvasElement[];
    for (let i = 0; i < Math.min(origCanvases.length, cloneCanvases.length); i++) {
      const srcCanvas = origCanvases[i];
      const dstCanvas = cloneCanvases[i];
      try {
        const dataUrl = srcCanvas.toDataURL('image/png');
        const img = document.createElement('img');
        img.src = dataUrl;
        // Preserve sizing from the original canvas element
        img.style.width = (dstCanvas.getAttribute('style')?.match(/width:[^;]+/i)?.[0]?.split(':')[1] || '100%').trim();
        img.style.height = (dstCanvas.getAttribute('style')?.match(/height:[^;]+/i)?.[0]?.split(':')[1] || 'auto').trim();
        img.style.display = 'block';
        dstCanvas.parentNode?.replaceChild(img, dstCanvas);
      } catch (e) {
        // If conversion fails, leave the canvas; most browsers will still print it
      }
    }

    // Print-specific styles (hide footer actions, basic fonts/margins)
    const printStyles = `
      <style>
        @page { margin: 16mm; }
        body { font-family: Arial, Helvetica, sans-serif; color: #111; }
        #result-footer { display: none !important; }
        img { max-width: 100%; }
      </style>`;

    const win = window.open('', '_blank');
    if (!win) { alert('Popup blocked. Please allow popups and try again.'); return; }
    win.document.open();
    win.document.write(`<!DOCTYPE html><html><head><meta charset="utf-8">${printStyles}</head><body>${clone.outerHTML}</body></html>`);
    win.document.close();

    const ensureImagesLoaded = () => {
      const imgs = win.document.images;
      let loaded = 0; const total = imgs.length;
      if (total === 0) { win.focus(); win.print(); return; }
      for (let i = 0; i < total; i++) {
        const img = imgs[i] as HTMLImageElement;
        if (img.complete) {
          loaded++;
          if (loaded === total) { win.focus(); win.print(); }
        } else {
          img.addEventListener('load', () => { loaded++; if (loaded === total) { win.focus(); win.print(); } });
          img.addEventListener('error', () => { loaded++; if (loaded === total) { win.focus(); win.print(); } });
        }
      }
    };
    setTimeout(ensureImagesLoaded, 100);
  }

  getStageProbability(stage: string): number {
  if (!this.classNames || !this.probabilities) return 0;
  // Use the normalized stage-to-index mapping
  const idxMap = this.classIndexByStage;
  const idx = idxMap[stage];
  if (idx === undefined || idx === null || idx < 0) return 0;
  return (this.probabilities[idx] ?? 0) * 100;
  }

  // Map model class labels to clinical stage names
  get clinicalStageMap(): { [key: string]: string } {
    return {
      'NonDemented': 'Non Demented',
      'Nondemented': 'Non Demented',
  'nondemented': 'Non Demented',
      'non demented': 'Non Demented',
  'non-demented': 'Non Demented',
      'VeryMildDemented': 'Very Mild Dementia',
      'Very Mild Demented': 'Very Mild Dementia',
      'very mild dementia': 'Very Mild Dementia',
  'very mild demented': 'Very Mild Dementia',
  'verymilddemented': 'Very Mild Dementia',
      'MildDemented': 'Mild Dementia',
      'Mild Demented': 'Mild Dementia',
      'mild dementia': 'Mild Dementia',
  'milddemented': 'Mild Dementia',
      'ModerateDemented': 'Moderate Dementia',
      'Moderate Demented': 'Moderate Dementia',
      'moderate dementia': 'Moderate Dementia',
  'moderatedemented': 'Moderate Dementia',
    };
  }

  get clinicalStageOrder(): string[] {
    return [
      'Non Demented',
      'Very Mild Dementia',
      'Mild Dementia',
      'Moderate Dementia'
    ];
  }

  // Subset requested for display (only first three stages)
  get limitedStageOrder(): string[] {
    return [
      'Non Demented',
      'Very Mild Dementia',
      'Mild Dementia'
    ];
  }

  get mappedClassNames(): string[] {
    if (!this.classNames) return [];
    return this.classNames.map(c => this.clinicalStageMap[c] || c);
  }

  get classIndexByStage(): { [key: string]: number } {
    const map: { [key: string]: number } = {};
    if (!this.classNames) return map;
    this.classNames.forEach((c, i) => {
      const original = (c ?? '').toString();
      const trimmed = original.trim();
      const lower = original.trim().toLowerCase();
      // Try exact key, then lowercase key, then match against canonical order case-insensitively
      let canonical = this.clinicalStageMap[original]
        || this.clinicalStageMap[trimmed]
        || this.clinicalStageMap[lower]
        || this.clinicalStageOrder.find(s => s.toLowerCase() === lower)
        || original;
      map[canonical] = i;
    });
    return map;
  }

  downloadImages() {
    const orig = this.normalizeUrl(this.downloadLinks?.original_image);
    if (orig) {
      const originalImageLink = document.createElement('a');
      originalImageLink.href = orig;
      originalImageLink.download = 'original_mri.png';
      originalImageLink.click();
    }

    const grad = this.normalizeUrl(this.gradCamUrl);
    if (grad) {
      const gradCamLink = document.createElement('a');
      gradCamLink.href = grad;
      gradCamLink.download = 'grad_cam_overlay.png';
      gradCamLink.click();
    }
  }

  ngOnInit() {
    this.gradCamUrl = this.normalizeUrl(this.gradCamUrl);
    if (this.downloadLinks) {
      this.downloadLinks.original_image = this.normalizeUrl(this.downloadLinks.original_image);
    }

    if (!this.gradCamUrl) {
      console.warn('Grad-CAM URL is missing or invalid.');
    }
    if (!this.downloadLinks?.original_image) {
      console.warn('Original image URL is missing or invalid.');
    }

    console.log('Grad-CAM URL:', this.gradCamUrl);
  console.log('Original Image URL:', this.downloadLinks?.original_image);
  }
}