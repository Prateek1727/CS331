import { Routes, Route } from 'react-router-dom'
import { DataProvider } from './context/DataContext'
import Sidebar from './components/Sidebar'
import Topbar from './components/Topbar'
import Dashboard from './pages/Dashboard'
import Tickets from './pages/Tickets'
import AIBrain from './pages/AIBrain'
import DecisionEngine from './pages/DecisionEngine'
import ActionLayer from './pages/ActionLayer'
import Intelligence from './pages/Intelligence'
import TicketDetail from './pages/TicketDetail'
import CustomerPortal from './pages/CustomerPortal'

export default function App() {
  return (
    <div className="app-layout">
      <Sidebar />
      <div className="main-wrapper">
        <Topbar />
        <main className="main-content">
          <DataProvider>
            <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/tickets" element={<Tickets />} />
            <Route path="/ai-brain" element={<AIBrain />} />
            <Route path="/decision-engine" element={<DecisionEngine />} />
            <Route path="/actions" element={<ActionLayer />} />
            <Route path="/intelligence" element={<Intelligence />} />
            <Route path="/ticket/:id" element={<TicketDetail />} />
            <Route path="/customer-portal" element={<CustomerPortal />} />
            </Routes>
          </DataProvider>
        </main>
      </div>
    </div>
  )
}
