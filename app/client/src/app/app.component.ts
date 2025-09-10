import { Component } from '@angular/core';
import { Router, RouterModule } from '@angular/router';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class App {
  constructor(private router: Router) {}

  openLogin() {
    this.router.navigate(['/login']);
  }

  openRegister() {
    this.router.navigate(['/register']);
  }
}
