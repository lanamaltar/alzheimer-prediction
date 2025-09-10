import { Component, Input, AfterViewInit, ViewChild, ElementRef, ChangeDetectionStrategy, OnChanges, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';

@Component({
  selector: 'app-numeric-prediction-result',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './numeric-prediction-result.component.html',
  styleUrls: ['./numeric-prediction-result.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class NumericPredictionResultComponent implements AfterViewInit, OnChanges {
  @Input() modelInternals: {
    coefficients?: { name: string, value: number }[],
    zScores?: { name: string, value: number }[],
    logitMargin?: number
  } = {};

  @Input() modelCalibration: {
    confidenceIntervalLow?: number,
    confidenceIntervalHigh?: number
  } = {};

  @Input() reliabilityScore: number|null = null;
  @ViewChild('barChartCanvas', { static: false }) barChartCanvas?: ElementRef<HTMLCanvasElement>;

  @Input() featureContributions: { name: string, contribution: number, percent_share: number }[] = [];
  @Input() featurePercentiles: { name: string, percentile: number }[] = [];
  @Input() shapValues: any = null;
  @Input() probabilityThreshold: number|null = null;
  @Input() calibrationNote: string|null = null;
  @Input() lowConfidenceFlag: boolean|null = null;
  @Input() featureCommentary: { name: string, comment: string }[] = [];
  ngAfterViewInit() {
    this.renderBarChart();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['probabilities']) {
      console.log('Probabilities updated:', this.probabilities);
    }
    if (changes['features']) {
      console.log('Features updated:', this.features);
    }
    if (changes['featureImportance']) {
      console.log('Feature Importance updated:', this.featureImportance);
    }
    if (changes['shapValues']) {
      console.log('SHAP Values updated:', this.shapValues);
    }
    if (changes['featureContributions']) {
      console.log('Feature Contributions updated:', this.featureContributions);
    }
    if (changes['featurePercentiles']) {
      console.log('Feature Percentiles updated:', this.featurePercentiles);
    }
    if (changes['featureCommentary']) {
      console.log('Feature Commentary updated:', this.featureCommentary);
    }
    if (changes['modelInternals']) {
      console.log('Model Internals updated:', this.modelInternals);
    }
    if (changes['reliabilityScore']) {
      console.log('Reliability Score updated:', this.reliabilityScore);
    }
    if (changes['timestamp']) {
      console.log('Timestamp updated:', this.timestamp);
    }
    this.renderBarChart();
    // If you use OnPush, inject ChangeDetectorRef and call markForCheck() here if needed
  }

  renderBarChart() {
    if (!this.barChartCanvas || !this.featureContributions?.length) return;

    const canvas = this.barChartCanvas.nativeElement as HTMLCanvasElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // ----- HiDPI scaling -----
    const dpr = Math.max(1, window.devicePixelRatio || 1);
  const cssWidth = canvas.clientWidth || 760;
  const cssHeight = canvas.clientHeight || 420;
    canvas.width = Math.floor(cssWidth * dpr);
    canvas.height = Math.floor(cssHeight * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // draw in CSS pixels

    // ----- Data (top 5) -----
    // Filter out feature contributions with zero percent_share
    const filteredContributions = this.featureContributions.filter(fc => fc.percent_share > 0);
    const data = filteredContributions.slice(0, 5);
    // Percent shares expected in [0,100]; sanitize
    const values = data.map(d => this.clamp(Number(d.percent_share) || 0, 0, 100));
    const labels = data.map(d => String(d.name));

    // ----- Layout -----
  const P = { top: 24, right: 24, bottom: 70, left: 52 };
  const chartW = cssWidth - P.left - P.right;
  const chartH = cssHeight - P.top - P.bottom;

  // ----- Y axis scale: fixed at 0,25,50,75,100 -----
  const yTicksArr = [0, 25, 50, 75, 100];
  const yMax = 100;

    // ----- Clear -----
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    // ----- Fonts & colors -----
    const font = '13px "Segoe UI", Arial, sans-serif';
    ctx.font = font;
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#333';

    // ----- Grid + Y ticks (more space, fixed at 0,20,40,60,80,100) -----
    ctx.strokeStyle = '#e6e6e6';
    ctx.lineWidth = 1;
    for (let i = 0; i < yTicksArr.length; i++) {
      const v = yTicksArr[i];
      const t = v / yMax;
      const y = P.top + chartH - t * chartH;
      // grid line
      ctx.beginPath();
      ctx.moveTo(P.left, y);
      ctx.lineTo(P.left + chartW, y);
      ctx.stroke();
      // tick label
      ctx.fillStyle = '#666';
      ctx.textAlign = 'right';
      ctx.fillText(`${v}%`, P.left - 8, y);
    }

    // ----- Axes -----
    ctx.strokeStyle = '#bdbdbd';
    ctx.beginPath();
    // y axis
    ctx.moveTo(P.left, P.top);
    ctx.lineTo(P.left, P.top + chartH);
    // x axis
    ctx.moveTo(P.left, P.top + chartH);
    ctx.lineTo(P.left + chartW, P.top + chartH);
    ctx.stroke();

    // ----- Bars (uniform color) -----
    const BAR_COLOR = '#b39ddb';       // requested fill
    const BAR_STROKE = '#7e57c2';      // subtle outline
    const barCount = values.length;
    const slot = chartW / Math.max(1, barCount);
  const barWidth = Math.min(60, Math.max(24, slot * 0.5));
  const gap = Math.max(8, slot - barWidth);

    for (let i = 0; i < barCount; i++) {
      const xCenter = P.left + (slot * i) + slot / 2;
      const v = values[i];
      const h = (v / yMax) * chartH;
      const x = xCenter - barWidth / 2;
      const y = P.top + chartH - h;

      // bar
      ctx.fillStyle = BAR_COLOR;
      ctx.fillRect(x, y, barWidth, h);

      // thin outline for crispness
      ctx.strokeStyle = BAR_STROKE;
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, y + 0.5, barWidth - 1, h - 1);

      // value label: inside if tall enough, else above
      ctx.font = 'bold 13px "Segoe UI", Arial, sans-serif';
      const valueText = `${v.toFixed(1)}%`;
      ctx.textAlign = 'center';
      if (h >= 22) {
        ctx.fillStyle = '#fff';
        ctx.fillText(valueText, xCenter, y + 12);
      } else {
        ctx.fillStyle = '#222';
        ctx.fillText(valueText, xCenter, y - 10);
      }

      // x labels (rotate slightly if long)
      const name = labels[i];
      ctx.save();
      ctx.translate(xCenter, P.top + chartH + 18);
      ctx.rotate(name.length > 8 ? -Math.PI / 8 : 0); // ~-22.5°
      ctx.fillStyle = '#333';
      ctx.font = font;
      ctx.textAlign = 'center';
      ctx.fillText(name, 0, 0);
      ctx.restore();

      // Accessible text (optional): add offscreen <span> if needed via ARIA elsewhere
    }

    // ----- Y axis title (optional) -----
    // ctx.save();
    // ctx.translate(16, P.top + chartH / 2);
    // ctx.rotate(-Math.PI / 2);
    // ctx.fillStyle = '#555';
    // ctx.font = '12px "Segoe UI", Arial, sans-serif';
    // ctx.textAlign = 'center';
    // ctx.fillText('Contribution (%)', 0, 0);
    // ctx.restore();
  }
  /**
   * Clamp a number between min and max (inclusive).
   */
  clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value));
  }

  /**
   * TypeScript-safe isNaN for template use.
   */
  isNaN(value: any): boolean {
    return typeof value === 'number' && Number.isNaN(value);
  }

  /**
   * Returns true if the page is being printed (for print styles/actions).
   */
  get isPrinting(): boolean {
    // This is a stub; you can enhance with media query listeners if needed.
    return false;
  }
  getFeatureBarWidth(weight: number): number {
    const max: number = this.maxFeatureWeight ?? 1;
    if (!max || max === 0) return 0;
    return Math.abs(weight) / max * 100;
  }

  getTopFeaturesString(): string {
    return (this.topFeatures && Array.isArray(this.topFeatures)) ? this.topFeatures.join(', ') : (this.topFeatures ? this.topFeatures.toString() : '');
  }
  @Input() result: number|null = null;
  @Input() probabilities: number[]|null = null;
  @Input() predictionLabel: string|null = null;
  @Input() features: number[]|null = null;
  @Input() featureNames: string[]|null = [
    'MR Delay','CDR','M/F', 'Age', 'EDUC','SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
  ;
  @Input() featureImportance: { name: string, weight: number }[] = [];
  @Input() predictionId: string = '';
  @Input() timestamp: string = '';

  // Derived and helper properties
  cdrScale = [0, 0.5, 1, 2, 3];
  cdrMeanings: { [key: number]: string } = {
  0: '0 = No Dementia',
  0.5: '0.5 = Very Mild Alzheimer’s disease',
  1: '1 = Mild Alzheimer’s disease',
  2: '2 = Moderate Alzheimer’s disease',
  3: '3 = Severe Alzheimer’s disease'
  };
  get confidenceTier(): string {
    if (!this.probabilities) return '';
    const p = Math.max(this.probabilities[0] || 0, this.probabilities[1] || 0);
    if (p >= 0.8) return 'High';
    if (p >= 0.65) return 'Medium';
    return 'Low';
  }
  get relativeTimeTooltip(): string {
    if (!this.timestamp) return '';
    const date = new Date(this.timestamp);
    const now = new Date();
    const diff = Math.floor((now.getTime() - date.getTime()) / 1000);
    if (diff < 60) return `${diff} seconds ago`;
    if (diff < 3600) return `${Math.floor(diff/60)} minutes ago`;
    if (diff < 86400) return `${Math.floor(diff/3600)} hours ago`;
    return `${Math.floor(diff/86400)} days ago`;
  }
  get maxFeatureWeight(): number {
    if (!this.featureImportance.length) return 1;
    return Math.max(...this.featureImportance.map(f => Math.abs(f.weight)));
  }
  get topFeatures(): string {
    if (!this.featureImportance.length) return '';
    return this.featureImportance.slice(0,3).map(f => f.name).join(', ');
  }
  get filteredFeatureContributions() {
    return this.featureContributions.filter(fc => fc.contribution !== 0);
  }
  get filteredShapValues() {
    return this.shapValues?.map((shapRow: number[]) => shapRow.filter(value => value !== 0)) || [];
  }
  getFeatureUserValue(name: string): string {
    if (!this.features || !this.featureNames) return '';
    const i = this.featureNames.indexOf(name);
    if (i === -1) return '';
    return this.formatFeatureValue(i, this.features[i]);
  }
  formatFeatureValue(i: number, value: any): string {
    if (i === 2) return value === 0 ? 'Female' : 'Male';
    if (i === 4) return value + ' years';
    if (i === 6) return value + '/30';
    if (i === 7) return value + ' mm³';
    if (i === 8) return value;
    if (i === 9) return value;
    return value;
  }

  copyId(id: string) {
    if (!navigator.clipboard) return;
    navigator.clipboard.writeText(id);
  }

  // Text-preserving export via browser's Print to PDF
  printResultCardViaBrowser() {
    const card = document.querySelector('.form-card.result-page-container') as HTMLElement | null;
    if (!card) {
      alert('Result card not found.');
      return;
    }

    // Clone the card
    const clone = card.cloneNode(true) as HTMLElement;

    // Replace canvases in clone with images copied from original canvases
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
        #footer-actions { display: none !important; }
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
        const img = imgs[i];
        if ((img as any).complete) {
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

  // Legacy image-based download kept for reference; not wired to UI
  async printResultCardAsPDF() {
    const card = document.querySelector('.form-card.result-page-container') as HTMLElement;
    if (!card) { alert('Result card not found.'); return; }
    const footerActions = document.getElementById('footer-actions');
    if (footerActions) footerActions.style.display = 'none';
    const canvas = await html2canvas(card, { scale: 2 });
    const imgData = canvas.toDataURL('image/png');
    const jsPdfInstance = new jsPDF({
      orientation: canvas.width > canvas.height ? 'landscape' : 'portrait',
      unit: 'px', format: [canvas.width, canvas.height]
    });
    jsPdfInstance.addImage(imgData, 'PNG', 0, 0, canvas.width, canvas.height);
    jsPdfInstance.save('prediction-result-card.pdf');
    if (footerActions) footerActions.style.display = '';
  }

  printResult() {
    window.print();
  }

  goBack() {
    window.location.href = '/predict/numeric';
  }


  abs(x: number): number {
    return Math.abs(x);
  }

  // ===== CDR-based stage explanation =====
  private getCdrValue(): number | null {
    if (!this.features || !this.featureNames) return null;
    const idx = this.featureNames.indexOf('CDR');
    if (idx === -1) return null;
    const v = this.features[idx];
    const n = typeof v === 'number' ? v : Number(v);
    return Number.isFinite(n) ? n : null;
  }

  private mapCdrToStage(cdr: number): { value: number, stage: string } {
    // Snap to nearest standard CDR level
    const levels = [0, 0.5, 1, 2, 3];
    let nearest = levels[0];
    let d = Infinity;
    for (const lv of levels) {
      const delta = Math.abs(cdr - lv);
      if (delta < d) { d = delta; nearest = lv; }
    }
    const mapping: Record<number, string> = {
      0: 'No Dementia',
      0.5: 'Very Mild Alzheimer’s disease',
      1: 'Mild Alzheimer’s disease',
      2: 'Moderate Alzheimer’s disease',
      3: 'Severe Alzheimer’s disease'
    } as any;
    return { value: nearest, stage: mapping[nearest as 0|0.5|1|2|3] || 'Unknown' };
  }

  getCdrAlzExplanation(): string {
    const cdr = this.getCdrValue();
    const adProb = (this.probabilities && this.probabilities[1] != null) ? (this.probabilities[1] * 100) : null;
    if (cdr == null || adProb == null) return '';
    const { value, stage } = this.mapCdrToStage(cdr);
    const shortStage = this.getShortStageLabel(stage);
    const stageSuffix = shortStage === 'No Dementia' ? '' : ' stage';

    if (adProb >= 60) {
      return `The model estimates an ${adProb.toFixed(0)}% probability of Alzheimer’s disease. Functional severity is consistent with ${shortStage}${stageSuffix} (CDR = ${value}).`;
    }
    if (adProb >= 20) {
      return `The model indicates a possible Alzheimer’s disease (${adProb.toFixed(0)}% probability). Functional severity is consistent with ${shortStage}${stageSuffix} (CDR = ${value}).`;
    }
    return `The model estimates a low probability of Alzheimer’s disease (${adProb.toFixed(0)}%). Functional severity is consistent with ${shortStage}${stageSuffix} (CDR = ${value}).`;
  }

  /**
   * Produce a concise stage label for readability (e.g., "Very Mild" from
   * "Very Mild Alzheimer’s disease"). "No Dementia" stays as-is.
   */
  private getShortStageLabel(stage: string): string {
    if (!stage) return '';
    const s = stage.trim();
    if (/^no dementia$/i.test(s)) return 'No Dementia';
    return s.replace(/\s*Alzheimer[’']s disease\s*$/i, '').trim();
  }
}