import { Component, OnInit, AfterViewInit, ChangeDetectorRef, HostListener } from '@angular/core';
import { ActivatedRoute, Router, NavigationEnd } from '@angular/router';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { ApiService } from '../api.service';
import { filter } from 'rxjs/operators';

interface HistoryItem {
  id: string;
  input_type: 'image' | 'numeric' | string;
  result?: string | number;
  predictionId?: string;
  confidence?: number;
  date?: string;
  gradCamUrl?: string;
  origUrl?: string;
  probabilities?: Record<string, number> | number[] | null;
  classNames?: string[];
  // Present for numeric predictions: raw input values in server order
  features?: any[];
}

@Component({
  selector: 'app-history',
  standalone: true,
  imports: [CommonModule, HttpClientModule],
  templateUrl: './history.component.html',
  styleUrls: ['./history.component.scss']
})
export class HistoryComponent implements OnInit, AfterViewInit {
  items: HistoryItem[] = [];
  imageItems: HistoryItem[] = [];
  numericItems: HistoryItem[] = [];
  nextCursor: string | null = null; // global fallback
  nextCursorImage: string | null = null;
  nextCursorNumeric: string | null = null;
  loading = false;
  loadingImage = false;
  loadingNumeric = false;
  error: string | null = null;
  private backendBase = 'http://localhost:5000';
  private retryCount = 0;

  // Lightbox state
  lightboxOpen = false;
  lightboxSrc: string | null = null;
  lightboxAlt: string = '';

  // Feature names for numeric inputs (must match server order)
  featureNames: string[] = ['MR Delay','CDR','M/F','Age','EDUC','SES','MMSE','eTIV','nWBV','ASF'];

  constructor(private api: ApiService, private router: Router, private cdr: ChangeDetectorRef, private route: ActivatedRoute) {}

  ngOnInit(): void {
    // Seed from resolver if available
    const resolved = this.route.snapshot.data?.['preload'];
    if (resolved?.success && Array.isArray(resolved.data?.history)) {
      this.items = resolved.data.history;
  this.nextCursor = resolved.data?.nextCursor ?? null;
  // When seeded globally, initialize type cursors equally for now
  this.nextCursorImage = this.nextCursor;
  this.nextCursorNumeric = this.nextCursor;
      this.error = null;
  // Initialize partitions
  const all = this.items || [];
  this.imageItems = all.filter(i => (i.input_type || '').toLowerCase() === 'image');
  this.numericItems = all.filter(i => (i.input_type || '').toLowerCase() === 'numeric');
    } else if (resolved?.message) {
      this.error = resolved.message;
    }

    // Attempt live load only if resolver didn't provide items
    if (!this.items || this.items.length === 0) {
      this.load();
    }
    // In case of reuse or initial quirks, re-trigger on NavigationEnd to /history
    this.router.events
      .pipe(filter((e): e is NavigationEnd => e instanceof NavigationEnd))
      .subscribe((e: NavigationEnd) => {
        if (e.urlAfterRedirects.endsWith('/history') && this.items.length === 0 && !this.loading) {
          this.load();
        }
      });
  }

  ngAfterViewInit(): void {
    // Fallback: if nothing rendered, retry shortly (handles rare timing issues)
    setTimeout(() => {
      if (this.items.length === 0 && !this.loading) {
        this.load();
      }
    }, 0);
  }

  // Close lightbox on Escape
  @HostListener('document:keydown.escape')
  onEscape() {
    if (this.lightboxOpen) {
      this.closeLightbox();
    }
  }

  private getToken(): string | null {
    // Primary key saved by login.component.ts
    const jwt = localStorage.getItem('jwt_token');
    if (jwt) return jwt;

    // Common alternates
    const direct = localStorage.getItem('token');
    if (direct) return direct;

    const raw = localStorage.getItem('auth');
    if (raw) {
      try {
        const parsed = JSON.parse(raw);
        return parsed?.token || parsed?.data?.token || null;
      } catch {
        return null;
      }
    }
    return null;
  }

  load(loadMore = false) {
    if (this.loading) return; // avoid duplicate calls
    const token = this.getToken();
    if (!token) { this.error = 'Nema tokena'; this.cdr.detectChanges(); return; }
    this.loading = true; this.cdr.detectChanges();
  this.api.getHistory(token, 20, loadMore ? this.nextCursor ?? undefined : undefined)
      .subscribe({
        next: (res) => {
          if (res?.success) {
            const history = res.data?.history ?? [];
            this.items = loadMore ? [...this.items, ...history] : history;
            // Partition into MRI (image) and numeric
            const all = this.items;
            this.imageItems = all.filter(i => (i.input_type || '').toLowerCase() === 'image');
            this.numericItems = all.filter(i => (i.input_type || '').toLowerCase() === 'numeric');
            this.nextCursor = res.data?.nextCursor ?? null;
            // When loading all types, sync type cursors
            this.nextCursorImage = this.nextCursor;
            this.nextCursorNumeric = this.nextCursor;
            this.error = null;
            this.retryCount = 0;
          } else {
            this.error = res?.message || 'Greška';
          }
          this.loading = false;
          this.cdr.detectChanges();
        },
        error: (err) => {
          // Handle rate limiting with a short, limited retry
          if (err?.status === 429 && this.retryCount < 2) {
            this.error = 'Previše zahtjeva, pokušavam ponovno…';
            this.retryCount++;
            this.loading = false;
            setTimeout(() => this.load(loadMore), 1200);
            this.cdr.detectChanges();
            return;
          }
          this.error = err?.error?.message || 'Greška pri dohvaćanju povijesti';
          this.retryCount = 0;
          this.loading = false;
          this.cdr.detectChanges();
        }
      });
  }

  loadMoreImage() {
    if (this.loadingImage) return;
    const token = this.getToken();
    if (!token) { this.error = 'Nema tokena'; this.cdr.detectChanges(); return; }
    this.loadingImage = true; this.cdr.detectChanges();
    this.api.getHistory(token, 20, this.nextCursorImage ?? undefined, 'image').subscribe({
      next: (res) => {
        if (res?.success) {
          const history = res.data?.history ?? [];
          // append only image items
          this.imageItems = [...this.imageItems, ...history];
          // also update global items for completeness
          this.items = [...this.items, ...history];
          this.nextCursorImage = res.data?.nextCursor ?? null;
          this.error = null;
          this.retryCount = 0;
        } else {
          this.error = res?.message || 'Greška';
        }
        this.loadingImage = false; this.cdr.detectChanges();
      },
      error: (err) => {
        if (err?.status === 429 && this.retryCount < 2) {
          this.error = 'Previše zahtjeva (MRI), pokušavam ponovno…';
          this.retryCount++;
          this.loadingImage = false;
          setTimeout(() => this.loadMoreImage(), 1200);
          this.cdr.detectChanges();
          return;
        }
        this.error = err?.error?.message || 'Greška pri dohvaćanju MRI povijesti';
        this.retryCount = 0;
        this.loadingImage = false; this.cdr.detectChanges();
      }
    });
  }

  loadMoreNumeric() {
    if (this.loadingNumeric) return;
    const token = this.getToken();
    if (!token) { this.error = 'Nema tokena'; this.cdr.detectChanges(); return; }
    this.loadingNumeric = true; this.cdr.detectChanges();
    this.api.getHistory(token, 20, this.nextCursorNumeric ?? undefined, 'numeric').subscribe({
      next: (res) => {
        if (res?.success) {
          const history = res.data?.history ?? [];
          this.numericItems = [...this.numericItems, ...history];
          this.items = [...this.items, ...history];
          this.nextCursorNumeric = res.data?.nextCursor ?? null;
          this.error = null;
          this.retryCount = 0;
        } else {
          this.error = res?.message || 'Greška';
        }
        this.loadingNumeric = false; this.cdr.detectChanges();
      },
      error: (err) => {
        if (err?.status === 429 && this.retryCount < 2) {
          this.error = 'Previše zahtjeva (Numeric), pokušavam ponovno…';
          this.retryCount++;
          this.loadingNumeric = false;
          setTimeout(() => this.loadMoreNumeric(), 1200);
          this.cdr.detectChanges();
          return;
        }
        this.error = err?.error?.message || 'Greška pri dohvaćanju numeric povijesti';
        this.retryCount = 0;
        this.loadingNumeric = false; this.cdr.detectChanges();
      }
    });
  }

  getUrl(u?: string | null): string | null {
    if (!u) return null;
    if (u.startsWith('http')) return u;
    if (u.startsWith('/')) return this.backendBase + u;
    return u;
  }

  getProb(it: HistoryItem, cls: string): number | null {
    const probs: any = it.probabilities as any;
    if (!probs) return null;
    if (Array.isArray(probs)) {
      if (it.classNames && Array.isArray(it.classNames)) {
        const idx = it.classNames.indexOf(cls);
        if (idx >= 0 && idx < probs.length) {
          const val = probs[idx];
          return typeof val === 'number' ? val : null;
        }
      }
      return null;
    }
    if (typeof probs === 'object') {
      const val = probs[cls];
      return typeof val === 'number' ? val : null;
    }
    return null;
  }

  // Lightbox controls
  openLightbox(src?: string | null, alt: string = '') {
    const url = this.getUrl(src ?? null);
    if (!url) return;
    this.lightboxSrc = url;
    this.lightboxAlt = alt || 'image';
    this.lightboxOpen = true;
    this.cdr.detectChanges();
  }

  closeLightbox() {
    this.lightboxOpen = false;
    this.lightboxSrc = null;
    this.lightboxAlt = '';
    this.cdr.detectChanges();
  }

  // ===== Binary label display mapping (history); keeps backend values intact =====
  mapBinaryLabel(label: string | number | undefined | null): string {
    if (label === undefined || label === null) return '-';
    const raw = String(label).trim();
    const low = raw.toLowerCase();
    // Normalize apostrophes
    const norm = low.replace(/[’]/g, "'");

    // Positive (Alzheimer's disease) aliases
    const adAliases = new Set([
      'demented', 'dementia', 'alzheimers', "alzheimer's", 'alzheimer’s', 'alzheimer', 'ad', '1'
    ]);
    if (adAliases.has(norm)) return 'Alzheimer’s disease';

    // Negative (Normal) aliases
    const normalAliases = new Set([
      'nondemented', 'non-demented', 'non demented', 'nondementia', 'non-dementia', 'non dementia', 'normal', 'control', 'healthy', '0'
    ]);
    if (normalAliases.has(norm)) return 'Normal';

    // Fallback: title case the raw label
    return raw.charAt(0).toUpperCase() + raw.slice(1);
  }

  // Format numeric feature values similarly to numeric result component
  formatFeatureValue(i: number, value: any): string {
    try {
      if (i === 2) return value === 0 ? 'Female' : 'Male';
      if (i === 4) return `${value} years`;
      if (i === 6) return `${value}/30`;
      if (i === 7) return `${value} mm³`;
      return String(value);
    } catch {
      return String(value);
    }
  }

  // ===== MRI rule-based label (reuse logic from image result) =====
  private clinicalStageMapDict(): { [key: string]: string } {
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

  private clinicalStageOrder(): string[] {
    return ['Non Demented', 'Very Mild Dementia', 'Mild Dementia', 'Moderate Dementia'];
  }

  // Display subset (omit Moderate Dementia) per request
  private limitedStageOrder(): string[] {
    return ['Non Demented', 'Very Mild Dementia', 'Mild Dementia'];
  }

  private classIndexByStageForItem(it: HistoryItem): { [key: string]: number } {
    const map: { [key: string]: number } = {};
    if (!it.classNames) return map;
    const stageMap = this.clinicalStageMapDict();
    const order = this.clinicalStageOrder();
    it.classNames.forEach((c, i) => {
      const original = (c ?? '').toString();
      const trimmed = original.trim();
      const lower = trimmed.toLowerCase();
      const canonical = stageMap[original]
        || stageMap[trimmed]
        || stageMap[lower]
        || order.find(s => s.toLowerCase() === lower)
        || original;
      map[canonical] = i;
    });
    return map;
  }

  getStageProbabilityForItem(it: HistoryItem, stage: string): number {
    if (!it.classNames || !it.probabilities) return 0;
    const idxMap = this.classIndexByStageForItem(it);
    const idx = idxMap[stage];
    if (idx === undefined || idx === null || idx < 0) return 0;
    const probs: any = it.probabilities as any;
    if (Array.isArray(probs)) {
      const v = probs[idx];
      return typeof v === 'number' ? (v * 100) : 0;
    }
    if (typeof probs === 'object') {
      const key = it.classNames[idx];
      const v = (probs as Record<string, number>)[key];
      return typeof v === 'number' ? (v * 100) : 0;
    }
    return 0;
  }

  private getDementiaCombinedProbabilityForItem(it: HistoryItem): number {
    const stages = ['Very Mild Dementia', 'Mild Dementia', 'Moderate Dementia'];
    return stages.reduce((acc, s) => acc + this.getStageProbabilityForItem(it, s), 0);
  }

  getDementiaRuleLabel(it: HistoryItem): string | null {
    if (!it.classNames || !it.probabilities) return null;
    const nonDem = this.getStageProbabilityForItem(it, 'Non Demented');
    const demSum = this.getDementiaCombinedProbabilityForItem(it);
    if (nonDem >= 80) return "Low risk of early Alzheimer’s";
    if (demSum >= 60) return "High likelihood of early Alzheimer’s";
    if (demSum >= 20) return "Possible early Alzheimer’s";
    return null;
  }

  // ===== Numeric (tabular) prediction CDR-based explanation (reuse logic from numeric result) =====
  private getNumericCdrValue(it: HistoryItem): number | null {
    if (!it?.features || !Array.isArray(it.features)) return null;
    const idx = this.featureNames.indexOf('CDR');
    if (idx === -1 || idx >= it.features.length) return null;
    const raw = it.features[idx];
    const n = typeof raw === 'number' ? raw : Number(raw);
    return Number.isFinite(n) ? n : null;
  }

  private mapCdrToStage(cdr: number): { value: number, stage: string } {
    const levels = [0, 0.5, 1, 2, 3];
    let nearest = levels[0]; let d = Infinity;
    for (const lv of levels) { const delta = Math.abs(cdr - lv); if (delta < d) { d = delta; nearest = lv; } }
    const mapping: Record<number, string> = {
      0: 'No Dementia',
      0.5: 'Very Mild Alzheimer’s disease',
      1: 'Mild Alzheimer’s disease',
      2: 'Moderate Alzheimer’s disease',
      3: 'Severe Alzheimer’s disease'
    } as any;
    return { value: nearest, stage: mapping[nearest as 0|0.5|1|2|3] || 'Unknown' };
  }

  private getShortStageLabel(stage: string): string {
    if (!stage) return '';
    const s = stage.trim();
    if (/^no dementia$/i.test(s)) return 'No Dementia';
    return s.replace(/\s*Alzheimer[’']s disease\s*$/i, '').trim();
  }

  getNumericExplanation(it: HistoryItem): string {
    if (!it || !it.classNames || !it.probabilities) return '';
    // probabilities may be array or object; assume binary ordering same as numeric result component (index 1 = AD)
    let adProb: number | null = null;
    const probs: any = it.probabilities as any;
    if (Array.isArray(probs)) {
      if (probs.length >= 2) adProb = typeof probs[1] === 'number' ? probs[1] * 100 : null;
    } else if (typeof probs === 'object') {
      // Find AD label by matching known positive aliases in classNames
      if (it.classNames.length >= 2) {
        const adIndex = it.classNames.findIndex(c => /dement|alzheimer/i.test(c));
        if (adIndex >= 0) {
          const key = it.classNames[adIndex];
          const val = probs[key];
          if (typeof val === 'number') adProb = val * 100;
        }
      }
    }
    if (adProb == null) return '';
    const cdr = this.getNumericCdrValue(it);
    if (cdr == null) return '';
    const { value, stage } = this.mapCdrToStage(cdr);
    const shortStage = this.getShortStageLabel(stage);
    const stageSuffix = shortStage === 'No Dementia' ? '' : ' stage';
    if (adProb >= 60) {
      return `The model estimates an ${adProb.toFixed(0)}% probability of Alzheimer’s disease. Functional severity is consistent with ${shortStage}${stageSuffix}.`;
    }
    if (adProb >= 20) {
      return `The model indicates a possible Alzheimer’s disease (${adProb.toFixed(0)}% probability). Functional severity is consistent with ${shortStage}${stageSuffix}.`;
    }
    return `The model estimates a low probability of Alzheimer’s disease (${adProb.toFixed(0)}%). Functional severity is consistent with ${shortStage}${stageSuffix}.`;
  }
}
