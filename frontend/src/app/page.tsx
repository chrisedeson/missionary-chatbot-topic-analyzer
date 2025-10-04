"use client";

import { useState, useEffect } from "react";
import { LogIn, LogOut } from "lucide-react";

import { Button } from "@/components/ui/button";
import { SidebarTrigger, SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { Dashboard } from "@/components/dashboard";
import { authManager } from "@/lib/auth";
import type { AuthState } from "@/lib/auth";

export default function Home() {
  const [authState, setAuthState] = useState<AuthState>({
    isAuthenticated: false,
    role: null,
    loading: true
  });
  const [loginPassword, setLoginPassword] = useState("");
  const [loginError, setLoginError] = useState("");
  const [showLogin, setShowLogin] = useState(false);

  useEffect(() => {
    const unsubscribe = authManager.subscribe(setAuthState);
    return unsubscribe;
  }, []);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoginError("");
    
    try {
      await authManager.login(loginPassword);
      setLoginPassword("");
      setShowLogin(false);
    } catch (error) {
      setLoginError(error instanceof Error ? error.message : "Login failed");
    }
  };

  const handleLogout = async () => {
    try {
      await authManager.logout();
    } catch (error) {
      console.error("Logout error:", error);
    }
  };

  if (authState.loading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="text-lg">Loading...</div>
      </div>
    );
  }

  return (
    <SidebarProvider>
      <div className="flex h-screen w-full">
        <AppSidebar isDeveloper={authState.isAuthenticated} />
        
        <div className="flex flex-col flex-1 min-w-0">
          <header className="flex items-center justify-between px-6 py-4 border-b sticky top-0 bg-background z-10">
            <div className="flex items-center gap-4">
              <SidebarTrigger />
              <div>
                <h1 className="text-xl font-semibold">BYU Pathway Topic Analyzer</h1>
                <p className="text-sm text-muted-foreground">
                  Student Question Analytics Dashboard
                </p>
              </div>
            </div>
          
          <div className="flex items-center gap-2">
            {authState.isAuthenticated ? (
              <Button
                variant="outline"
                size="sm"
                onClick={handleLogout}
              >
                <LogOut className="w-4 h-4 mr-2" />
                Sign Out
              </Button>
            ) : (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowLogin(!showLogin)}
                >
                  <LogIn className="w-4 h-4 mr-2" />
                  Sign in as Developer
                </Button>
                
                {showLogin && (
                  <form onSubmit={handleLogin} className="flex gap-2">
                    <input
                      type="password"
                      placeholder="Developer Password"
                      value={loginPassword}
                      onChange={(e) => setLoginPassword(e.target.value)}
                      className="px-3 py-1 text-sm border rounded"
                      autoFocus
                    />
                    <Button type="submit" size="sm">
                      Login
                    </Button>
                  </form>
                )}
              </>
            )}
          </div>
        </header>
        
        {loginError && (
          <div className="mx-6 mt-4 p-3 bg-destructive/10 border border-destructive/20 rounded-md">
            <p className="text-sm text-destructive">{loginError}</p>
          </div>
        )}
        
        <main className="flex-1 overflow-auto p-6 md:p-8">
          <div className="max-w-7xl mx-auto">
            <Dashboard isDeveloper={authState.isAuthenticated} />
          </div>
        </main>
      </div>
    </div>
    </SidebarProvider>
  );
}
