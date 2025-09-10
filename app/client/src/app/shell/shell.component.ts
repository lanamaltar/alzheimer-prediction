import { Component, ChangeDetectorRef, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet, RouterLink, Router } from '@angular/router';
import { AuthService } from '../auth.service';

@Component({
  selector: 'app-shell',
  standalone: true,
  imports: [CommonModule, RouterOutlet, RouterLink],
  templateUrl: './shell.component.html',
  styleUrls: ['./shell.component.scss']
})
export class ShellComponent {
  showLogoutModal = false;
  sidebarOpen = true;
  userName = '';
  private readonly mobileBreakpoint = 768; // px

  constructor(private cdr: ChangeDetectorRef, private router: Router, private auth: AuthService) {
    this.userName = this.auth.getUserName();
    // Initialize sidebar state based on initial window size
    this.setSidebarResponsive(window.innerWidth);
  }

  @HostListener('window:resize', ['$event'])
  onWindowResize(event: UIEvent) {
    const width = (event.target as Window).innerWidth;
    this.setSidebarResponsive(width);
  }

  private setSidebarResponsive(width: number) {
    // Auto-close on small screens; do not force-open on larger screens to respect user toggle
    if (width < this.mobileBreakpoint && this.sidebarOpen) {
      this.sidebarOpen = false;
      this.cdr.detectChanges();
    }
  }

  openLogoutModal() {
    this.showLogoutModal = true;
    this.cdr.detectChanges();
  }

  closeLogoutModal() {
    this.showLogoutModal = false;
    this.cdr.detectChanges();
  }

  confirmLogout() {
    this.showLogoutModal = false;
    this.cdr.detectChanges();
    // Perform logout logic here
    console.log('User logged out');
  this.auth.logout();
    // Navigate to the root path (landing page)
    this.router.navigate(['']);
  }

  toggleSidebar() {
    this.sidebarOpen = !this.sidebarOpen;
    this.cdr.detectChanges();
  }
}
