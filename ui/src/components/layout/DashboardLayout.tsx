import React from 'react';
import { Navbar, Sidebar, Footer } from './components';
import { Box, Container } from '@/components/ui';

interface DashboardLayoutProps {
  children: React.ReactNode;
}

export const DashboardLayout: React.FC<DashboardLayoutProps> = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <Box className="min-h-screen bg-gray-100">
      {/* ナビゲーションバー */}
      <Navbar onMenuClick={() => setSidebarOpen(true)} />

      {/* サイドバー */}
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      {/* メインコンテンツ */}
      <Box
        component="main"
        className="flex-1 py-6 px-4 sm:px-6 lg:px-8"
      >
        {children}
      </Box>

      {/* フッター */}
      <Footer />
    </Box>
  );
};

// ナビゲーションバーコンポーネント
const Navbar: React.FC<{ onMenuClick: () => void }> = ({ onMenuClick }) => {
  return (
    <nav className="bg-white shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <button
              onClick={onMenuClick}
              className="px-4 text-gray-500 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-indigo-500 md:hidden"
            >
              <span className="sr-only">Open menu</span>
              <MenuIcon className="h-6 w-6" />
            </button>
            <div className="flex-shrink-0 flex items-center">
              <img
                className="h-8 w-auto"
                src="/logo.svg"
                alt="AI Ops Platform"
              />
            </div>
          </div>
          <div className="flex items-center">
            <UserMenu />
          </div>
        </div>
      </div>
    </nav>
  );
};

// サイドバーコンポーネント
const Sidebar: React.FC<{ open: boolean; onClose: () => void }> = ({
  open,
  onClose
}) => {
  const navigation = [
    { name: 'Dashboard', href: '/', icon: HomeIcon },
    { name: 'Incidents', href: '/incidents', icon: AlertIcon },
    { name: 'Services', href: '/services', icon: ServerIcon },
    { name: 'Analysis', href: '/analysis', icon: ChartIcon },
    { name: 'Settings', href: '/settings', icon: SettingsIcon },
  ];

  return (
    <div
      className={`fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-lg transform transition-transform duration-300 ease-in-out ${
        open ? 'translate-x-0' : '-translate-x-full'
      }`}
    >
      <div className="h-full flex flex-col">
        <div className="h-16 flex items-center justify-between px-4">
          <span className="text-lg font-semibold">AI Ops Platform</span>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
          >
            <XIcon className="h-6 w-6" />
          </button>
        </div>
        <nav className="flex-1 px-4 space-y-1">
          {navigation.map((item) => (
            <a
              key={item.name}
              href={item.href}
              className="flex items-center px-2 py-2 text-sm font-medium text-gray-600 rounded-md hover:bg-gray-50 hover:text-gray-900"
            >
              <item.icon className="mr-3 h-6 w-6" />
              {item.name}
            </a>
          ))}
        </nav>
      </div>
    </div>
  );
};

// フッターコンポーネント
const Footer: React.FC = () => {
  return (
    <footer className="bg-white border-t border-gray-200">
      <Container>
        <div className="py-4 text-center text-sm text-gray-500">
          © 2024 AI Ops Platform. All rights reserved.
        </div>
      </Container>
    </footer>
  );
};