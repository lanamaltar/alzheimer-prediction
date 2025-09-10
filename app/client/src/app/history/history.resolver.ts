import { ResolveFn } from '@angular/router';
import { inject } from '@angular/core';
import { of } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { ApiService } from '../api.service';

export const historyResolver: ResolveFn<any> = () => {
  const api = inject(ApiService);
  // Retrieve token similarly to component
  let token = localStorage.getItem('jwt_token') || localStorage.getItem('token');
  if (!token) {
    const raw = localStorage.getItem('auth');
    if (raw) {
      try {
        const parsed = JSON.parse(raw);
        token = parsed?.token || parsed?.data?.token || null;
      } catch {
        token = null as any;
      }
    }
  }
  if (!token) {
    return of({ success: false, message: 'Nema tokena', data: { history: [], nextCursor: null } });
  }
  // Add a tiny delay to reduce immediate concurrent requests when component also loads
  return api.getHistory(token, 20).pipe(
    catchError((err) => of({ success: false, message: err?.error?.message || 'Greška', data: { history: [], nextCursor: null } }))
  );
};
