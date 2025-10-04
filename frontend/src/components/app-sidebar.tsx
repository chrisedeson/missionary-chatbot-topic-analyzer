"use client";

import { BarChart3, Upload, Settings, HelpCircle, Database } from "lucide-react";

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  SidebarFooter,
} from "@/components/ui/sidebar";

interface AppSidebarProps {
  isDeveloper: boolean;
}

export function AppSidebar({ isDeveloper }: AppSidebarProps) {
  const navigationItems = [
    {
      title: "Dashboard",
      url: "#",
      icon: BarChart3,
      active: true,
    },
  ];

  const developerItems = isDeveloper ? [
    {
      title: "Upload Data",
      url: "#upload",
      icon: Upload,
    },
    {
      title: "Database",
      url: "#database",
      icon: Database,
    },
    {
      title: "Settings",
      url: "#settings",
      icon: Settings,
    },
  ] : [];

  const helpItems = [
    {
      title: "Help & Support",
      url: "#help",
      icon: HelpCircle,
    },
  ];

  return (
    <Sidebar>
      <SidebarHeader className="border-b">
        <div className="px-2 py-2">
          <h2 className="text-lg font-semibold">Topic Analyzer</h2>
          <p className="text-xs text-muted-foreground">
            Student Question Analytics
          </p>
        </div>
      </SidebarHeader>
      
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {navigationItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild isActive={item.active}>
                    <a href={item.url}>
                      <item.icon className="w-4 h-4" />
                      <span>{item.title}</span>
                    </a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {isDeveloper && (
          <SidebarGroup>
            <SidebarGroupLabel>Developer Tools</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {developerItems.map((item) => (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton asChild>
                      <a href={item.url}>
                        <item.icon className="w-4 h-4" />
                        <span>{item.title}</span>
                      </a>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        )}
        
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              {helpItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild>
                    <a href={item.url}>
                      <item.icon className="w-4 h-4" />
                      <span>{item.title}</span>
                    </a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      
      <SidebarFooter className="border-t">
        <div className="px-2 py-2">
          <p className="text-xs text-muted-foreground">
            BYU-Pathway Worldwide
          </p>
          <p className="text-xs text-muted-foreground">
            {isDeveloper ? "Developer Mode" : "View Only"}
          </p>
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}