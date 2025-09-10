import { Component, ElementRef, EventEmitter, HostListener, Input, Output, ViewChild, OnChanges, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';

export interface SparklineChartDataItem<V = number> {
  id: string | number;
  x: number; // days from 0
  y: V;      // CDR
}

interface CircleRenderItem {
  id: string | number;
  cx: number;
  cy: number;
  radius: number;
  data: SparklineChartDataItem;
}

@Component({
  selector: 'app-sparkline-chart',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './sparkline-chart.component.html',
  styleUrls: ['./sparkline-chart.component.scss']
})
export class SparklineChartComponent implements OnChanges {
  @Input() idProperty: keyof SparklineChartDataItem = 'id';
  @Input() xValue: keyof SparklineChartDataItem = 'x';
  @Input() yValue: keyof SparklineChartDataItem = 'y';
  @Input() data: SparklineChartDataItem[] = [];
  @Input() radius = 5;
  // Graph decorations
  @Input() xLabel: string = 'MR Delay (days)';
  @Input() yLabel: string = 'CDR';
  @Input() xTickCount: number = 4;
  @Input() yTickCount: number = 4;
  @Input() margin: { top: number; right: number; bottom: number; left: number } = { top: 10, right: 10, bottom: 34, left: 40 };
  @Input() xMin?: number;
  @Input() xMax?: number;
  @Input() yTicksOverride?: number[];
  @Input() xTicksOverride?: number[];

  @Output() dataPointClicked = new EventEmitter<SparklineChartDataItem>();
  @Output() dataPointHover = new EventEmitter<SparklineChartDataItem | null>();

  @ViewChild('svgContainer', { static: false, read: ElementRef }) svgContainer!: ElementRef<HTMLDivElement>;

  width = 600;
  height = 220;
  linePath = '';
  circles: CircleRenderItem[] = [];
  // Axes metrics and ticks
  axisX = 40; // left
  axisY = 200; // bottom
  xTicks: Array<{ x: number; label: string }> = [];
  yTicks: Array<{ y: number; label: string }> = [];

  constructor(private host: ElementRef) {}

  ngAfterViewInit() {
    // initial render
    setTimeout(() => this.render());
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['data'] || changes['radius'] || changes['padding']) {
      // Render on next tick to ensure container has dimensions
      Promise.resolve().then(() => this.render());
    }
  }

  @HostListener('window:resize')
  onResize() { this.render(); }

  private render() {
    if (!this.svgContainer?.nativeElement) return;
    const el = this.svgContainer.nativeElement as HTMLDivElement;
    const hostEl = this.host.nativeElement as HTMLElement;
    // Width fallbacks
    let w = el.clientWidth;
    if (!w) w = hostEl.clientWidth;
    if (!w && hostEl.parentElement) w = hostEl.parentElement.clientWidth;
    if (!w) {
      const rect = el.getBoundingClientRect();
      w = rect.width || 600;
    }
    // Height fallbacks
    let h = el.clientHeight;
    if (!h) {
      const comp = getComputedStyle(el);
      const parsed = parseFloat(comp.height);
      h = isFinite(parsed) && parsed > 0 ? parsed : 220;
    }
    this.width = w;
    this.height = h;

    // If layout not ready (very small), retry shortly
    if (this.width < 40 || this.height < 40) {
      setTimeout(() => this.render(), 60);
      return;
    }

    const m = this.margin;
    const innerW = Math.max(0, this.width - m.left - m.right);
    const innerH = Math.max(0, this.height - m.top - m.bottom);

  const xs = this.data.map((d) => Number(d[this.xValue]));
  const ys = this.data.map((d) => Number(d[this.yValue]));
    let xMin = this.xMin ?? Math.min(...xs, 0);
    let xMax = this.xMax ?? Math.max(...xs, 1);
    if (this.xTicksOverride && this.xTicksOverride.length) {
      const oMin = Math.min(...this.xTicksOverride);
      const oMax = Math.max(...this.xTicksOverride);
      xMin = Math.min(xMin, oMin);
      xMax = Math.max(xMax, oMax);
    }
    let yMin = Math.min(...ys, 0);
    let yMax = Math.max(...ys, 1);
    if (this.yTicksOverride && this.yTicksOverride.length) {
      const oMin = Math.min(...this.yTicksOverride);
      const oMax = Math.max(...this.yTicksOverride);
      yMin = Math.min(yMin, oMin);
      yMax = Math.max(yMax, oMax);
    }
    if (yMin === yMax) { yMin -= 1; yMax += 1; }
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;

    const xScale = (x: number) => m.left + ((x - xMin) / xRange) * innerW;
    const yScale = (y: number) => m.top + (innerH - ((y - yMin) / yRange) * innerH);

    // Axis positions
    this.axisX = m.left;
    this.axisY = m.top + innerH;

    // Build path in x order: M x0 y0 L x1 y1 ...
    if (this.data.length > 0) {
      const ordered = [...this.data].sort((a, b) => Number(a[this.xValue]) - Number(b[this.xValue]));
      const parts: string[] = [];
      ordered.forEach((d, i) => {
        const x = xScale(Number(d[this.xValue]));
        const y = yScale(Number(d[this.yValue]));
        parts.push(i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`);
      });
      this.linePath = parts.join(' ');
    } else {
      this.linePath = '';
    }

    this.circles = this.data.map((d) => {
      const cx = xScale(Number(d[this.xValue]));
      const cy = yScale(Number(d[this.yValue]));
      return {
        id: d[this.idProperty] as string | number,
        cx,
        cy,
        radius: this.radius,
        data: d,
      };
    });

    // Ticks
    const xVals = (this.xTicksOverride && this.xTicksOverride.length)
      ? this.xTicksOverride
      : this.niceTicks(xMin, xMax, Math.max(2, this.xTickCount));
    this.xTicks = xVals.map((v) => ({ x: xScale(v), label: this.formatTick(v) }));
  const yVals = this.yTicksOverride && this.yTicksOverride.length ? this.yTicksOverride : this.niceTicks(yMin, yMax, Math.max(2, this.yTickCount));
  this.yTicks = yVals.map((v) => ({ y: yScale(v), label: this.formatTick(v) }));
  }

    circleTrackBy = (_: number, entity: CircleRenderItem) => entity.id;
  onDataPointClicked(d: SparklineChartDataItem) { this.dataPointClicked.emit(d); }
  onDataPointHover(d: SparklineChartDataItem | null) { this.dataPointHover.emit(d); }

  private niceTicks(min: number, max: number, count: number): number[] {
    if (!isFinite(min) || !isFinite(max) || count <= 0) return [];
    if (min === max) return [min];
    const step = (max - min) / count;
    const ticks: number[] = [];
    for (let i = 0; i <= count; i++) ticks.push(min + step * i);
    return ticks;
  }

  private formatTick(v: number): string {
    // 0.5 steps look cleaner with up to 2 decimals
    const abs = Math.abs(v);
    return abs >= 100 ? v.toFixed(0) : abs >= 10 ? v.toFixed(1) : v.toFixed(2);
  }
}