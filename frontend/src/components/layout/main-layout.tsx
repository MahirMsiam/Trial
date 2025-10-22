import { ReactNode } from 'react';
import Footer from './footer';
import Header from './header';

interface MainLayoutProps {
  children: ReactNode;
  currentPage: 'search' | 'chat' | 'about';
}

export default function MainLayout({ children, currentPage }: MainLayoutProps) {
  return (
    <div className="min-h-screen flex flex-col">
      <Header currentPage={currentPage} />
      <main className="flex-1 container mx-auto px-4 py-8">{children}</main>
      <Footer />
    </div>
  );
}
