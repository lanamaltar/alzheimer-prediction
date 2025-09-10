import { Component } from '@angular/core';
import { ApiService } from '../api.service';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.scss']
})
export class LoginComponent {
  username = '';
  password = '';
  message = '';
  success = false;

  constructor(private apiService: ApiService, private router: Router) {}


  onLogin() {
    this.apiService.login({ username: this.username, password: this.password }).subscribe({
      next: (res) => {
        console.log('Login response:', res);
        this.success = res.success;
        this.message = res.message || 'Login successful!';
        if (res.success && res.data && res.data.token) {
          localStorage.setItem('jwt_token', res.data.token);
          localStorage.setItem('userName', this.username);
          this.username = '';
          this.password = '';
          // Debug log before navigation
          console.log('Login successful, navigating to /dashboard');
          // Redirect to dashboard after successful login
          this.router.navigate(['/dashboard']);
        }
      },
      error: (err) => {
        this.success = false;
        this.message = err.error?.message || 'Login failed. Please try again.';
      }
    });
  }
}
