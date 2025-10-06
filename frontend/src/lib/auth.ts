import { apiClient } from './api'

export interface AuthState {
  isAuthenticated: boolean
  role: string | null
  loading: boolean
}

export class AuthManager {
  private listeners: ((state: AuthState) => void)[] = []
  private state: AuthState = {
    isAuthenticated: false,
    role: null,
    loading: true
  }

  constructor() {
    // Check for existing auth token on initialization
    if (typeof window !== 'undefined') {
      const token = localStorage.getItem('dev_auth_token')
      if (token) {
        this.state = {
          isAuthenticated: true,
          role: 'developer',
          loading: false
        }
      } else {
        this.state.loading = false
      }
    }
  }

  subscribe(listener: (state: AuthState) => void) {
    this.listeners.push(listener)
    // Immediately call with current state
    listener(this.state)
    
    // Return unsubscribe function
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener)
    }
  }

  private notify() {
    this.listeners.forEach(listener => listener(this.state))
  }

  async login(password: string): Promise<void> {
    this.state.loading = true
    this.notify()

    try {
      const response = await apiClient.login(password)
      
      this.state = {
        isAuthenticated: true,
        role: response.role,
        loading: false
      }
      
      this.notify()
    } catch (error) {
      this.state.loading = false
      this.notify()
      throw error
    }
  }

  async logout(): Promise<void> {
    this.state.loading = true
    this.notify()

    try {
      await apiClient.logout()
      
      this.state = {
        isAuthenticated: false,
        role: null,
        loading: false
      }
      
      this.notify()
    } catch (error) {
      // Even if logout fails on server, clear local state
      this.state = {
        isAuthenticated: false,
        role: null,
        loading: false
      }
      
      this.notify()
    }
  }

  getState(): AuthState {
    return { ...this.state }
  }

  isDeveloper(): boolean {
    return this.state.isAuthenticated && this.state.role === 'developer'
  }
}

export const authManager = new AuthManager()
export default authManager