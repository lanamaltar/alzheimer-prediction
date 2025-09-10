// ...existing code...
import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private apiUrl = '/api'; 

  constructor(private http: HttpClient) {}

  register(data: { username: string; password: string }): Observable<any> {
    return this.http.post(`${this.apiUrl}/register`, data);
  }

  login(data: { username: string; password: string }): Observable<any> {
    return this.http.post(`${this.apiUrl}/login`, data);
  }

  predictNumeric(input: number[], token: string): Observable<any> {
    return this.http.post(
      `${this.apiUrl}/predict/numeric`,
      { input },
      { headers: new HttpHeaders({ Authorization: `Bearer ${token}` }) }
    );
  }

  predictImage(image: File, token: string): Observable<any> {
    const formData = new FormData();
    formData.append('image', image);
    return this.http.post(`${this.apiUrl}/predict/image`, formData, {
      headers: new HttpHeaders({ Authorization: `Bearer ${token}` }),
    });
  }

  getHistory(token: string, limit = 20, cursor?: string, type?: string): Observable<any> {
    let url = `${this.apiUrl}/history?limit=${limit}`;
    if (cursor) url += `&cursor=${encodeURIComponent(cursor)}`;
    if (type) url += `&type=${encodeURIComponent(type)}`;
    return this.http.get(url, {
      headers: new HttpHeaders({ Authorization: `Bearer ${token}` }),
    });
  }
  
}