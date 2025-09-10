// ...existing imports...
import { Component, ChangeDetectorRef } from '@angular/core';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../api.service';
import { NumericPredictionResultComponent } from '../numeric-prediction-result/numeric-prediction-result.component';

@Component({
  selector: 'app-predict-numeric',
  standalone: true,
  imports: [CommonModule, FormsModule, NumericPredictionResultComponent],
  templateUrl: './predict-numeric.component.html',
  styleUrls: ['./predict-numeric.component.scss']
})

export class PredictNumericComponent {
  // Advanced model internals and explainability fields
  featureContributions: { name: string, contribution: number, percent_share: number }[] = [];
  featurePercentiles: { name: string, percentile: number }[] = [];
  shapValues: any = null;
  probabilityThreshold: number|null = null;
  calibrationNote: string|null = null;
  lowConfidenceFlag: boolean|null = null;
  featureCommentary: { name: string, comment: string }[] = [];
  modelInternals: any = null;
  modelCalibration: any = null;
  reliabilityScore: number|null = null;
  focusedIndex: number|null = null;
  featuresInput: string[] = Array(10).fill('');
  featureNames: string[] = ['MR Delay','CDR','M/F', 'Age', 'EDUC','SES', 'MMSE', 'eTIV', 'nWBV', 'ASF'];
  featureExplanations: string[] = [
    'MR Delay Time (Contrast) (Note: In this model, higher MR Delay is associated with lower dementia risk; see tooltip for details.)',
    'Clinical Dementia Rating',
    'Gender (0 = Male, 1 = Female)',
    'Age in years',
    'Years of education',
    'Socio Economic Status',
    'Mini Mental State Examination',
    'Estimated Total Intracranial Volume',
    'Normalize Whole Brain Volume',
    'Atlas Scaling Factor'
  ];
  loading = false;
  result: number | null = null;
  probabilities: number[] | null = null;
  predictionLabel: string | null = null;
  featureImportance: { name: string, weight: number }[] = [];
  features: number[] | null = null;
  predictionId: string = '';
  timestamp: string = '';
  error: string | null = null;
  keyInfluencingFeatures: string[] = []; 

  constructor(private apiService: ApiService, private cdr: ChangeDetectorRef, private router: Router) {}

  trackByIndex(index: number, obj: any): any {
    return index;
  }

  onSubmit() {
    this.error = null;
    this.result = null;
    this.probabilities = null;
    this.predictionLabel = null;
    // Convert all feature values to numbers
    const numericFeatures = this.featuresInput.map(f => Number(f));
    console.log('Submitting features:', numericFeatures); // Debug log
    if (numericFeatures.some(f => isNaN(f))) {
      this.error = 'Please fill in all 10 features with valid numbers.';
      this.loading = false;
      console.error('Validation failed: not all features are numbers', numericFeatures); // Debug log
      this.cdr.detectChanges();
      return;
    }
    const token = localStorage.getItem('jwt_token');
    if (!token) {
      this.error = 'You must be logged in.';
      this.loading = false;
      console.error('No JWT token found'); // Debug log
      return;
    }
    this.loading = true;
    console.log('Sending request to /api/predict/numeric with token:', token); // Debug log
    this.apiService.predictNumeric(numericFeatures, token).subscribe({
      next: (res: any) => {
        console.log('API response:', res); // Debug log
        this.loading = false;
        if (res.success && res.data) {
          // Assign all result data for the result card
          this.result = res.data.prediction ?? null;
          this.probabilities = res.data.probabilities ?? null;
          this.predictionLabel = res.data.prediction_label ?? null;
          this.featureImportance = res.data.feature_importance ?? [];
          this.features = res.data.features ?? numericFeatures;
          this.predictionId = res.data.prediction_id ?? '';
          this.timestamp = res.data.timestamp ?? '';
          this.featureContributions = res.data.feature_contributions ?? [];
          this.featurePercentiles = res.data.feature_percentiles ?? [];
          // Validate SHAP values to ensure all values are numbers
          this.shapValues = (res.data.shapValues ?? res.data.shap_values ?? []).map((row: any) =>
            row.map((value: any) => (isNaN(Number(value)) ? 0 : Number(value)))
          );
          this.probabilityThreshold = res.data.probabilityThreshold ?? res.data.probability_threshold ?? null;
          this.calibrationNote = res.data.calibrationNote ?? res.data.calibration_note ?? null;
          this.lowConfidenceFlag = res.data.lowConfidenceFlag ?? res.data.low_confidence_flag ?? null;
          this.featureCommentary = res.data.feature_commentary ?? [];
          // --- Assign advanced fields if present ---
          // Normalize modelInternals and modelCalibration for both GET and POST shapes
          const internals = res.data.modelInternals ?? res.data.model_internals ?? res.data;
          const calibration = res.data.modelCalibration ?? res.data.model_calibration ?? res.data;
          // Adapt modelInternals to expected shape for template
          if (internals && (internals.coefficients || internals.feature_coefficients || internals.zScores || internals.z_scores || internals.logitMargin || internals.logit_margin)) {
            this.modelInternals = {
              coefficients: (internals.coefficients || internals.feature_coefficients || []).map((c: any) => ({
                name: c.name,
                value: c.coef
              })),
              zScores: (internals.zScores || internals.z_scores || []).map((z: any) => ({
                name: z.name,
                value: z.z
              })),
              logitMargin: internals.logitMargin ?? internals.logit_margin
            };
            // Derive reliabilityScore from logitMargin if present
            const logitMargin = this.modelInternals.logitMargin;
            if (typeof logitMargin === 'number') {
              this.reliabilityScore = 1 / (1 + Math.exp(-Math.abs(logitMargin)));
            } else {
              this.reliabilityScore = null;
            }
            // --- Build key influencing features: nonzero |z*coef|, sorted descending ---
            if (
              Array.isArray(this.modelInternals.coefficients) &&
              Array.isArray(this.modelInternals.zScores)
            ) {
              type KeyInfluence = { name: string; score: number };
              const keyInfluences: KeyInfluence[] = this.modelInternals.coefficients
                .map((coef: { name: string; value: number }, i: number): KeyInfluence => {
                  const z = this.modelInternals.zScores[i]?.value ?? 0;
                  // Only count if coef is nonzero
                  const score = Math.abs(z * (Math.abs(coef.value) >= 1e-6 ? coef.value : 0));
                  return { name: coef.name, score };
                })
                .filter((f: KeyInfluence) => f.score > 0)
                .sort((a: KeyInfluence, b: KeyInfluence) => b.score - a.score);
              const names = keyInfluences.map((f: KeyInfluence) => f.name);
              this.keyInfluencingFeatures = names.filter((v: string, i: number, arr: string[]) => v && arr.indexOf(v) === i);
            } else {
              this.keyInfluencingFeatures = [];
            }
          } else {
            this.modelInternals = null;
            this.reliabilityScore = null;
            this.keyInfluencingFeatures = [];
          }
          this.modelCalibration = calibration.modelCalibration ?? calibration.model_calibration ?? null;
          this.cdr.detectChanges();
        } else {
          this.error = res.message || 'Prediction failed.';
          this.cdr.detectChanges();
        }
      },
      error: (err: any) => {
        this.loading = false;
        this.error = err.error?.message || 'Prediction failed. Please try again.';
        console.error('API error:', err); // Debug log
        this.cdr.detectChanges();
      }
    });
  }

  get filteredModelCoefficients(): { name: string, value: number }[] {
    if (!this.modelInternals?.coefficients) return [];
    return this.modelInternals.coefficients.filter(
      (c: { name: string, value: number }) => Math.abs(c.value) >= 1e-6
    );
  }
}