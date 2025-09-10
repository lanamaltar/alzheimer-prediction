import { Component, ChangeDetectorRef } from '@angular/core';
import { ImagePredictResultComponent } from '../image-predict-result/image-predict-result.component';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-predict-image',
  standalone: true,
  imports: [CommonModule, FormsModule, ImagePredictResultComponent],
  templateUrl: './predict-image.component.html',
  styleUrls: ['./predict-image.component.scss']
})
export class PredictImageComponent {
  selectedFile: File | null = null;
  loading = false;
  error: string | null = null;
  result: any = null;

  constructor(private http: HttpClient, private cdr: ChangeDetectorRef) {}

  onFileChange(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length) {
      this.selectedFile = input.files[0];
      this.error = null;
      this.result = null;
    }
  }

  onSubmit() {
    if (!this.selectedFile) {
      this.error = 'Please select an MRI image file.';
      return;
    }
    this.loading = true;
    this.error = null;
    this.result = null;
    const formData = new FormData();
    formData.append('image', this.selectedFile);
    formData.append('token', localStorage.getItem('jwt_token') || '');

    this.http.post<any>('/api/predict/image', formData).subscribe({
      next: (data) => {
        console.log('Prediction response:', data);
        const cls = data?.data?.classNames as string[] | null;
        const probsRaw = data?.data?.probabilities;
        const probsMap = data?.data?.probabilitiesMap as Record<string, number> | undefined;
        let probs: number[] = Array.isArray(probsRaw) ? probsRaw : [];
        if ((!probs || !probs.length) && probsMap && Array.isArray(cls)) {
          // Build list from map in class order if needed
          probs = cls.map(name => typeof probsMap[name] === 'number' ? probsMap[name] : 0);
        }
        this.result = {
          prediction: data?.data?.prediction ?? null,
          probabilities: probs || [],
          classNames: cls || [],
          confidence: data?.data?.confidence,
          entropy: typeof data?.data?.entropy === 'number' ? data.data.entropy : (Number(data?.data?.entropy) || null),
          reliability: this.mapReliability(data?.data?.reliability),
          gradCamUrl: data?.data?.gradCamUrl || null,
          modelInfo: data?.data?.modelInfo || {},
          inputDetails: data?.data?.inputDetails || {},
          downloadLinks: data?.data?.downloadLinks || {},
          timestamp: data?.data?.timestamp || null,
          predictionId: data?.data?.predictionId || null
        };
        this.loading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Prediction error:', err);
        this.error = err?.error?.message || 'Prediction failed. Please try again.';
        this.loading = false;
        this.cdr.detectChanges();
      },
      complete: () => {
        console.log('Prediction request complete');
      }
    });
  }

  private mapReliability(reliability: string | null): string {
    switch (reliability) {
      case 'High':
        return 'Definitive';
      case 'Medium':
        return 'Moderate';
      case 'Low':
        return 'Not definitive';
      default:
        return 'Unknown';
    }
  }
}
